"""The CLI's pipeline subclass must not pin the sink to a fixed frame size.

The face app's aggregator emits the original frame at the *detector's* square
geometry (e.g. 640x640) while the stock ``_appsink`` caps to the requested
width/height (640x480).  videoconvert cannot rescale, so caps negotiation fails,
the aggregator blocks once its bypass queue fills and the pipeline silently
delivers a handful of frames and then nothing.

Pure string/source tests: no GStreamer, no hailo-apps import.
"""
import importlib.util
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bmo_face.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("bmo_face_cli_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bmo_face_cli_pipeline"] = module
    spec.loader.exec_module(module)
    return module


class FakeFaceReconPipeline:
    """Stands in for the face app's FaceReconPipeline."""

    def __init__(self, frames, faces, input_source="usb", width=640, height=480,
                 db_path="", **kwargs):
        self.frames = frames
        self.faces = faces
        self._input = input_source
        self.width = width
        self.height = height

    def _source(self):
        return "v4l2src ! jpegdec"

    def _appsink(self):
        return ("queue name=app_q ! videoconvert ! "
                f"video/x-raw,format=RGB,width={self.width},height={self.height} ! "
                "appsink name=web_appsink emit-signals=true sync=false")


class FakeModule:
    FaceReconPipeline = FakeFaceReconPipeline


def make(cli, **kwargs):
    return cli._pipeline_class(FakeModule)(None, None, input_source="rpi",
                                           width=640, height=480,
                                           db_path="/tmp/x.db", **kwargs)


def test_rpi_source_listens_on_the_given_port(cli):
    pipe = make(cli, udp_port=5099)
    source = pipe._source()
    assert "udpsrc port=5099" in source
    assert "avdec_h264" in source


def test_non_rpi_source_is_left_to_the_app(cli):
    pipe = cli._pipeline_class(FakeModule)(None, None, input_source="usb",
                                           width=640, height=480,
                                           db_path="/tmp/x.db", udp_port=5099)
    assert pipe._source() == "v4l2src ! jpegdec"


def test_appsink_does_not_pin_frame_size(cli):
    """A pinned size breaks negotiation with the aggregator's square output."""
    sink = make(cli, udp_port=5099)._appsink()
    assert "width=" not in sink, sink
    assert "height=" not in sink, sink


def test_appsink_keeps_the_name_the_app_looks_up(cli):
    """``pipeline.py`` resolves the sink by name to attach its frame callback."""
    assert "name=web_appsink" in make(cli, udp_port=5099)._appsink()


def test_appsink_stays_rgb_and_drops_stale_frames(cli):
    sink = make(cli, udp_port=5099)._appsink()
    assert "video/x-raw,format=RGB" in sink
    assert "drop=true" in sink
    assert "max-buffers=" in sink
