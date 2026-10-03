"""图片理解模块测试（纯函数，不调 VLM API）"""

import io

import pytest
from PIL import Image

from knowledge_pipeline.image_understanding import (
    is_worth_processing,
    _shrink_for_low_detail,
    VLM_LOW_DETAIL_MAX_SIDE,
    image_to_base64,
)


class TestIsWorthProcessing:
    """图片值不值得花一次 VLM 调用

    原实现是 `img.width * img.height * 3 > max_bytes * 10`：
    ① 比文档写的 2MB 宽松 10 倍；② w*h*3 是未压缩像素字节数，
    和 PNG 实际大小差一个数量级。
    """

    def test_rejects_tiny_images(self):
        assert is_worth_processing(Image.new("RGB", (80, 80))) is False
        assert is_worth_processing(Image.new("RGB", (600, 60))) is False

    def test_accepts_normal_slide(self):
        assert is_worth_processing(Image.new("RGB", (1600, 1200))) is True
        assert is_worth_processing(Image.new("RGB", (200, 200))) is True

    def test_size_limit_is_real_bytes_not_estimate(self):
        """按真实编码大小判断：纯色大图压缩后很小，应当通过"""
        solid = Image.new("RGB", (3000, 3000), "white")
        assert is_worth_processing(solid) is True
        buf = io.BytesIO()
        solid.save(buf, format="PNG")
        assert buf.tell() < 2 * 1024 * 1024

    def test_rejects_when_over_real_byte_budget(self):
        solid = Image.new("RGB", (3000, 3000), "white")
        buf = io.BytesIO()
        solid.save(buf, format="PNG")
        real = buf.tell()
        # 把上限压到比真实大小还低，必须被拒
        assert is_worth_processing(solid, max_bytes=real - 1) is False
        assert is_worth_processing(solid, max_bytes=real + 1) is True

    def test_min_size_is_configurable(self):
        img = Image.new("RGB", (200, 200))
        assert is_worth_processing(img, min_size=100) is True
        assert is_worth_processing(img, min_size=500) is False


class TestLowDetailShrinking:
    """detail="low" 原来只是个没人用的参数，现在真的会缩放图片"""

    def test_shrinks_large_image(self):
        img = Image.new("RGB", (4000, 2000))
        out = _shrink_for_low_detail(img)
        assert max(out.size) == VLM_LOW_DETAIL_MAX_SIDE
        # 保持宽高比
        assert out.width / out.height == pytest.approx(2.0, abs=0.01)

    def test_does_not_upscale_small_image(self):
        img = Image.new("RGB", (800, 600))
        assert _shrink_for_low_detail(img).size == (800, 600)

    def test_square_image(self):
        out = _shrink_for_low_detail(Image.new("RGB", (3000, 3000)))
        assert out.size == (VLM_LOW_DETAIL_MAX_SIDE, VLM_LOW_DETAIL_MAX_SIDE)

    def test_base64_roundtrip(self):
        import base64

        img = Image.new("RGB", (120, 90), "red")
        data = base64.b64decode(image_to_base64(img))
        assert Image.open(io.BytesIO(data)).size == (120, 90)


class TestVlmPayloadHonoursDetail:
    def test_low_and_high_upload_different_sizes(self, monkeypatch):
        import base64

        import knowledge_pipeline.image_understanding as iu

        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None, **kw):
            sent["payload"] = json

            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "描述"}}]}

            return R()

        monkeypatch.setattr(iu.requests, "post", fake_post)

        big = Image.new("RGB", (4000, 2000), "white")
        iu.describe_image_vlm(big, context="测试", detail="low")
        url = sent["payload"]["messages"][1]["content"][1]["image_url"]["url"]
        low_img = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        assert max(low_img.size) <= VLM_LOW_DETAIL_MAX_SIDE

        iu.describe_image_vlm(big, context="测试", detail="high")
        url = sent["payload"]["messages"][1]["content"][1]["image_url"]["url"]
        high_img = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        assert high_img.size == (4000, 2000)


class TestPdfProcessingSmoke:
    """两个处理器现在用 with 打开 PDF（异常也能释放句柄）

    这些 PDF 体积较大且未纳入版本控制，文件不在时跳过。
    """

    PDF = "data/操作系统/ppt/03-OS调度.pdf"

    def test_ppt_processor_runs(self):
        import os

        if not os.path.exists(self.PDF):
            pytest.skip("测试用 PDF 不在仓库中")
        from knowledge_pipeline.ppt_processor import process_ppt_pdf

        slides = process_ppt_pdf(self.PDF, use_vlm=False, max_pages=2)
        assert len(slides) == 2
        assert slides[0].page_num == 1

    def test_missing_pdf_raises_without_leaking(self):
        from knowledge_pipeline.ppt_processor import process_ppt_pdf

        with pytest.raises(Exception):
            process_ppt_pdf("data/definitely-not-here.pdf", use_vlm=False, max_pages=1)

    def test_should_use_vlm_matches_documented_rules(self):
        from knowledge_pipeline.ppt_processor import _should_use_vlm

        assert _should_use_vlm("字" * 200, "进程调度", 5) is False   # 中等文字量，保守跳过
        assert _should_use_vlm("字" * 100, "进程调度", 5) is True    # 文字极少
        assert _should_use_vlm("字" * 500, "进程调度", 5) is False   # 文字多
        assert _should_use_vlm("字" * 200, "流程图", 5) is True      # 标题含图表关键词
        assert _should_use_vlm("短", "封面", 1) is False             # 封面页
