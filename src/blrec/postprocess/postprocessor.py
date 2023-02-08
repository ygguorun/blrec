from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from pathlib import PurePath
from typing import Any, Awaitable, Dict, Final, Iterator, List, Optional, Tuple, Union

from reactivex.scheduler import ThreadPoolScheduler

from ..bili.live import Live
from ..core import Recorder, RecorderEventListener
from ..event.event_emitter import EventEmitter, EventListener
from ..exception import exception_callback, submit_exception
from ..flv.helpers import is_valid_flv_file
from ..flv.metadata_analysis import analyse_metadata
from ..flv.metadata_injection import InjectingProgress, inject_metadata
from ..logging.room_id import aio_task_with_room_id
from ..path import danmaku_path, extra_metadata_path, record_metadata_path
from ..utils.mixins import AsyncCooperationMixin, AsyncStoppableMixin, SupportDebugMixin
from .ffmpeg_metadata import make_metadata_file
from .helpers import copy_files_related, discard_dir, discard_file, get_extra_metadata
from .models import DeleteStrategy, PostprocessorStatus
from .remux import RemuxingProgress, RemuxingResult, remux_video
from .typing import Progress

from bcut_asr import BcutASR, APIError
from bcut_asr.orm import ResultStateEnum
import time
import ffmpeg

__all__ = (
    'Postprocessor',
    'PostprocessorEventListener',
    'PostprocessorStatus',
    'DeleteStrategy',
)


logger = logging.getLogger(__name__)


class PostprocessorEventListener(EventListener):
    async def on_video_postprocessing_completed(
        self, postprocessor: Postprocessor, path: str
    ) -> None:
        ...


class Postprocessor(
    EventEmitter[PostprocessorEventListener],
    RecorderEventListener,
    AsyncStoppableMixin,
    AsyncCooperationMixin,
    SupportDebugMixin,
):
    _worker_semaphore: Final = asyncio.Semaphore(value=1)

    def __init__(
        self,
        live: Live,
        recorder: Recorder,
        *,
        remux_to_mp4: bool = False,
        inject_extra_metadata: bool = False,
        delete_source: DeleteStrategy = DeleteStrategy.AUTO,
    ) -> None:
        super().__init__()
        self._init_for_debug(live.room_id)

        self._live = live
        self._recorder = recorder

        self.remux_to_mp4 = remux_to_mp4
        self.inject_extra_metadata = inject_extra_metadata
        self.delete_source = delete_source

        self._status = PostprocessorStatus.WAITING
        self._postprocessing_path: Optional[str] = None
        self._postprocessing_progress: Optional[Progress] = None
        self._completed_files: List[str] = []

    @property
    def recorder(self) -> Recorder:
        return self._recorder

    @property
    def status(self) -> PostprocessorStatus:
        return self._status

    @property
    def postprocessing_path(self) -> Optional[str]:
        return self._postprocessing_path

    @property
    def postprocessing_progress(self) -> Optional[Progress]:
        return self._postprocessing_progress

    def get_completed_files(self) -> Iterator[str]:
        yield from iter(self._completed_files)

    async def on_recording_started(self, recorder: Recorder) -> None:
        # clear completed files of previous recording
        self._completed_files.clear()

    async def on_video_file_completed(self, recorder: Recorder, path: str) -> None:
        self._queue.put_nowait(path)

    async def on_danmaku_file_completed(self, recorder: Recorder, path: str) -> None:
        self._completed_files.append(path)

    async def _do_start(self) -> None:
        self._recorder.add_listener(self)

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._scheduler = ThreadPoolScheduler()
        self._task = asyncio.create_task(self._worker())
        self._task.add_done_callback(exception_callback)

        logger.debug('Started postprocessor')

    async def _do_stop(self) -> None:
        self._recorder.remove_listener(self)

        await self._queue.join()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

        del self._queue
        del self._scheduler
        del self._task

        logger.debug('Stopped postprocessor')

    @aio_task_with_room_id
    async def _worker(self) -> None:

        while True:
            self._status = PostprocessorStatus.WAITING
            self._postprocessing_path = None
            self._postprocessing_progress = None

            video_path = await self._queue.get()

            async with self._worker_semaphore:
                logger.debug(f'Postprocessing... {video_path}')

                await self._wait_for_metadata_file(video_path)

                try:
                    if video_path.endswith('.flv'):
                        if not await self._is_vaild_flv_file(video_path):
                            logger.warning(f'The flv file may be invalid: {video_path}')
                        if self.remux_to_mp4:
                            self._status = PostprocessorStatus.REMUXING
                            (
                                result_path,
                                remuxing_result,
                            ) = await self._remux_video_to_mp4(video_path)
                            if not self._debug:
                                await discard_file(
                                    extra_metadata_path(video_path), 'DEBUG'
                                )
                                if self._should_delete_source_files(remuxing_result):
                                    await discard_file(video_path)
                        elif self.inject_extra_metadata:
                            self._status = PostprocessorStatus.INJECTING
                            result_path = await self._inject_extra_metadata(video_path)
                        else:
                            result_path = video_path

                    elif video_path.endswith('.m3u8'):
                        if self.remux_to_mp4:
                            self._status = PostprocessorStatus.REMUXING
                            (
                                result_path,
                                remuxing_result,
                            ) = await self._remux_video_to_mp4(video_path)
                            await copy_files_related(video_path)
                            if result_path != video_path:
                                self._completed_files.append(danmaku_path(result_path))
                                self._completed_files.remove(danmaku_path(video_path))
                            if not self._debug:
                                if self._should_delete_source_files(remuxing_result):
                                    await discard_dir(os.path.dirname(video_path))
                        else:
                            result_path = video_path
                    else:
                        result_path = video_path

                    self._completed_files.append(result_path)
                    await self._gen_video_subtitle(result_path)
                    await self._emit(
                        'video_postprocessing_completed', self, result_path
                    )
                except Exception as exc:
                    submit_exception(exc)
                finally:
                    self._queue.task_done()

    async def _inject_extra_metadata(self, path: str) -> str:
        logger.info(f"Injecting metadata for '{path}' ...")
        try:
            try:
                metadata = await get_extra_metadata(path)
            except Exception as e:
                logger.warning(f'Failed to get extra metadata: {repr(e)}')
                logger.info(f"Analysing metadata for '{path}' ...")
                await self._analyse_metadata(path)
                metadata = await get_extra_metadata(path)
            else:
                if 'keyframes' not in metadata:
                    logger.warning('The keyframes metadata lost')
                    logger.info(f"Analysing metadata for '{path}' ...")
                    await self._analyse_metadata(path)
                    new_metadata = await get_extra_metadata(path)
                    metadata.update(new_metadata)
            await self._inject_metadata(path, metadata)
        except Exception as e:
            logger.error(f"Failed to inject metadata for '{path}': {repr(e)}")
            submit_exception(e)
        else:
            logger.info(f"Successfully injected metadata for '{path}'")
        return path

    async def _remux_video_to_mp4(self, in_path: str) -> Tuple[str, RemuxingResult]:
        if in_path.endswith('.flv'):
            out_path = str(PurePath(in_path).with_suffix('.mp4'))
            metadata_path = await make_metadata_file(in_path)
        elif in_path.endswith('.m3u8'):
            out_path = str(PurePath(in_path).parent.with_suffix('.mp4'))
            metadata_path = await make_metadata_file(in_path)
        else:
            raise NotImplementedError(in_path)

        logger.info(f"Remuxing '{in_path}' to '{out_path}' ...")
        remux_result = await self._remux_video(in_path, out_path, metadata_path)

        if remux_result.is_failed():
            logger.error(f"Failed to remux '{in_path}' to '{out_path}'")
            result_path = in_path
        elif remux_result.is_warned():
            logger.warning('Remuxing done, but ran into problems.')
            result_path = out_path
        elif remux_result.is_successful():
            logger.info(f"Successfully remuxed '{in_path}' to '{out_path}'")
            result_path = out_path
        else:
            pass

        logger.debug(f'ffmpeg output:\n{remux_result.output}')

        if not self._debug and in_path.endswith('.flv'):
            await discard_file(metadata_path, 'DEBUG')

        return result_path, remux_result

    def _analyse_metadata(self, path: str) -> Awaitable[None]:
        future: asyncio.Future[None] = asyncio.Future()
        self._postprocessing_path = path

        subscription = analyse_metadata(path, show_progress=True).subscribe(
            on_error=lambda e: future.set_exception(e),
            on_completed=lambda: future.set_result(None),
            scheduler=self._scheduler,
        )
        future.add_done_callback(lambda f: subscription.dispose())

        return future

    def _inject_metadata(self, path: str, metadata: Dict[str, Any]) -> Awaitable[None]:
        future: asyncio.Future[None] = asyncio.Future()
        self._postprocessing_path = path

        def on_next(value: InjectingProgress) -> None:
            self._postprocessing_progress = value

        subscription = inject_metadata(path, metadata, show_progress=True).subscribe(
            on_next=on_next,
            on_error=lambda e: future.set_exception(e),
            on_completed=lambda: future.set_result(None),
            scheduler=self._scheduler,
        )
        future.add_done_callback(lambda f: subscription.dispose())

        return future

    def _remux_video(
        self, in_path: str, out_path: str, metadata_path: str
    ) -> Awaitable[RemuxingResult]:
        future: asyncio.Future[RemuxingResult] = asyncio.Future()
        self._postprocessing_path = in_path

        def on_next(value: Union[RemuxingProgress, RemuxingResult]) -> None:
            if isinstance(value, RemuxingProgress):
                self._postprocessing_progress = value
            elif isinstance(value, RemuxingResult):
                future.set_result(value)

        subscription = remux_video(
            in_path,
            out_path,
            metadata_path,
            show_progress=True,
            remove_filler_data=True,
        ).subscribe(
            on_next=on_next,
            on_error=lambda e: future.set_exception(e),
            scheduler=self._scheduler,
        )
        future.add_done_callback(lambda f: subscription.dispose())

        return future

    async def _is_vaild_flv_file(self, video_path: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, is_valid_flv_file, video_path)

    def _should_delete_source_files(self, remux_result: RemuxingResult) -> bool:
        if self.delete_source == DeleteStrategy.AUTO:
            if not remux_result.is_failed():
                return True
        elif self.delete_source == DeleteStrategy.SAFE:
            if not remux_result.is_failed() and not remux_result.is_warned():
                return True
        elif self.delete_source == DeleteStrategy.NEVER:
            return False

        return False

    async def _wait_for_metadata_file(self, video_path: str) -> None:
        loop = asyncio.get_running_loop()

        if video_path.endswith('.flv'):
            path = extra_metadata_path(video_path)
        elif video_path.endswith('.m3u8'):
            path = record_metadata_path(video_path)
        else:
            return

        for _ in range(10):
            if await loop.run_in_executor(None, os.path.isfile, path):
                break
            else:
                logger.debug(f'Not found metadata file: {path}')
                await asyncio.sleep(1)
        else:
            logger.warning(f'No such metadata file: {path}')

    async def _gen_video_subtitle(self, video_path: str) -> None:
        loop = asyncio.get_running_loop()

        OFFSET_DURATION = int(60 * 60 * 2)

        def ffmpeg_render(media_file: str, offset_start=-1) -> bytes:
            '提取视频伴音并转码为aac格式'
            media_info = ffmpeg.probe(media_file)
            if offset_start != -1:
                for stream in media_info['streams']:
                    if stream['codec_type'] == 'audio':
                        if stream['codec_name'] == 'aac':
                            out, err = (ffmpeg
                                        .input(media_file, v='warning')
                                        .output('pipe:', ss = offset_start, t = OFFSET_DURATION, acodec='copy', ac=1, format='adts')
                                        .run(capture_stdout=True)
                                        )
                        else:
                            out, err = (ffmpeg
                                        .input(media_file, v='warning')
                                        .output('pipe:', ss = offset_start, t = OFFSET_DURATION, acodec='copy', ac=1, format='adts')
                                        .run(capture_stdout=True)
                                        )
                        break
            else:
                for stream in media_info['streams']:
                    if stream['codec_type'] == 'audio':
                        if stream['codec_name'] == 'aac':
                            out, err = (ffmpeg
                                        .input(media_file, v='warning')
                                        .output('pipe:', acodec='copy', ac=1, format='adts')
                                        .run(capture_stdout=True)
                                        )
                        else:
                            out, err = (ffmpeg
                                        .input(media_file, v='warning')
                                        .output('pipe:', ac=1, format='adts')
                                        .run(capture_stdout=True)
                                        )
                        break
            return out

        def get_media_duration(media_file: str) -> int:
            '获取媒体文件的时长'
            media_info = ffmpeg.probe(media_file)
            duration = media_info['format']['duration']
            return int(duration.split('.')[0])


        # 处理输入文件情况
        infile_name = video_path

        # 处理输出文件情况
        outfile_fmt = 'srt'
        outfile_name = f"{infile_name.rsplit('.', 1)[-2]}.{outfile_fmt}"
        outfile = open(outfile_name, 'w', encoding='utf8')



        # ffmpeg分离视频伴音
        media_duration = get_media_duration(infile_name)
        logging.info('非标准音频文件, 尝试调用ffmpeg转码')
        result_list = []
        if media_duration > OFFSET_DURATION:
            logging.info(f'媒体时长过长，采取分段上传')

        for i in range((media_duration + OFFSET_DURATION - 1) // OFFSET_DURATION):
            logging.info(f'当前上传第 {i + 1} 段')
            try:
                infile_data = ffmpeg_render(infile_name, OFFSET_DURATION * i)
            except ffmpeg.Error:
                logging.error('ffmpeg转码失败')
            else:
                logging.info('ffmpeg转码完成')
                infile_fmt = 'aac'

            # 开始执行转换逻辑
            asr = BcutASR()
            asr.set_data(raw_data=infile_data, data_fmt=infile_fmt)
            try:
                # 上传文件
                asr.upload()
                # 创建任务
                task_id = asr.create_task()
                while True:
                    # 轮询检查任务状态
                    try:
                        task_resp = asr.result()
                        match task_resp.state:
                            case ResultStateEnum.STOP:
                                logging.info(f'等待识别开始')
                            case ResultStateEnum.RUNING:
                                logging.info(f'识别中-{task_resp.remark}')
                            case ResultStateEnum.ERROR:
                                logging.error(f'识别失败-{task_resp.remark}')
                                break
                            case ResultStateEnum.COMPLETE:
                                logging.info(f'识别成功')
                                # 识别成功, 回读字幕数据
                                result_part = task_resp.parse()
                                # result_part.adjust_timestamp_offset(i*OFFSET_DURATION)
                                # 调整字幕 offset
                                for item in result_part.utterances:
                                    item.start_time += i * OFFSET_DURATION * 1000
                                    item.end_time += i * OFFSET_DURATION * 1000

                                result_list.append(result_part)
                                break
                        time.sleep(5.0)
                    except Exception as e:
                        print(e)
                        time.sleep(5.0)
                
            except APIError as err:
                logging.error(f'接口错误: {err.__str__()}')
        result = result_list[0]
        for item in result_list[1:]:
            result.utterances.extend(item.utterances)


        if not result.has_data():
            logging.error('{video_path}未识别到语音')
        match outfile_fmt:
            case 'srt':
                outfile.write(result.to_srt())
            case 'lrc':
                outfile.write(result.to_lrc())
            case 'json':
                outfile.write(result.json())
            case 'txt':
                outfile.write(result.to_txt())
        logging.info(f'转换成功: {outfile_name}')
