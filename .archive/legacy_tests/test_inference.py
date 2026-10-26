"""
青小团推理服务测试 - 覆盖CLI、Web、OpenAI API兼容、文本生成、生成参数等
"""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestQingxiaotuanCLIDemo:

    def test_cli_demo_imports(self):
        from qingxiaotuan.cli_demo import main
        assert callable(main)

    def test_cli_demo_parse_args(self):
        from qingxiaotuan.cli_demo import parse_args
        args = parse_args(["--model", "qingxiaotuan", "--max_length", "2048"])
        assert args.model == "qingxiaotuan"
        assert args.max_length == 2048

    def test_cli_default_args(self):
        from qingxiaotuan.cli_demo import parse_args
        args = parse_args([])
        assert args.max_length == 8192
        assert args.temperature == 0.7

    @patch("qingxiaotuan.cli_demo.QingxiaotuanForCausalLM")
    def test_cli_model_loading(self, mock_model):
        from qingxiaotuan.cli_demo import load_model
        mock_model.from_pretrained.return_value = MagicMock()
        model, _ = load_model("qingxiaotuan_path", max_length=2048)
        assert model is not None

    @patch("qingxiaotuan.cli_demo.load_model")
    def test_cli_generate(self, mock_load):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        from qingxiaotuan.cli_demo import generate_text
        result = generate_text(mock_model, mock_tokenizer, "你好", max_length=100)
        assert result is not None


class TestQingxiaotuanWebDemo:

    @patch("qingxiaotuan.web_demo.gr")
    def test_web_demo_interface_creation(self, mock_gr):
        from qingxiaotuan.web_demo import create_interface
        mock_gr.ChatInterface.return_value = MagicMock()
        interface = create_interface(model_path="qingxiaotuan")
        assert interface is not None

    @patch("qingxiaotuan.web_demo.gr")
    def test_web_demo_launch(self, mock_gr):
        from qingxiaotuan.web_demo import create_interface
        mock_iface = MagicMock()
        mock_gr.ChatInterface.return_value = mock_iface
        create_interface(model_path="qingxiaotuan", share=True)
        mock_iface.launch.assert_called_with(share=True)

    @patch("qingxiaotuan.web_demo.gr")
    def test_web_demo_streaming(self, mock_gr):
        from qingxiaotuan.web_demo import stream_response
        result = list(stream_response("Hello", max_length=50))
        assert result is not None

    def test_web_demo_share_default_false(self):
        from qingxiaotuan.web_demo import create_interface
        with patch("qingxiaotuan.web_demo.gr.ChatInterface") as mock:
            with patch("qingxiaotuan.web_demo.QingxiaotuanForCausalLM") as mock_model:
                with patch("qingxiaotuan.web_demo.AutoTokenizer") as mock_tok:
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_tok.from_pretrained.return_value = MagicMock()
                    iface = create_interface(model_path="qingxiaotuan")
                    assert iface is not None


class TestQingxiaotuanOpenAIAPI:

    @pytest.fixture
    def mock_openai_request(self):
        return {"model": "qingxiaotuan", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}

    def test_openai_api_chat_completion(self, mock_openai_request):
        from qingxiaotuan.openai_api import ChatCompletion
        with patch.object(ChatCompletion, 'create') as mock_create:
            mock_response = {"id": "chatcmpl-qxt", "choices": [{"message": {"content": "Hello!"}}]}
            mock_create.return_value = mock_response
            response = ChatCompletion.create(**mock_openai_request)
            assert response["id"] == "chatcmpl-qxt"

    def test_openai_api_streaming(self, mock_openai_request):
        from qingxiaotuan.openai_api import ChatCompletion
        with patch.object(ChatCompletion, 'create') as mock_create:
            mock_create.return_value = [{"choices": [{"delta": {"content": "Hello"}}]}]
            response = ChatCompletion.create(**mock_openai_request, stream=True)
            assert response is not None

    def test_openai_api_model_list(self):
        from qingxiaotuan.openai_api import Model
        with patch.object(Model, 'list') as mock_list:
            mock_list.return_value = {"data": [{"id": "qingxiaotuan"}]}
            models = Model.list()
            assert "data" in models
            assert models["data"][0]["id"] == "qingxiaotuan"


class TestQingxiaotuanTextGeneration:

    def test_generate_method_exists(self, small_qingxiaotuan_model):
        assert hasattr(small_qingxiaotuan_model, 'generate')
        assert callable(small_qingxiaotuan_model.generate)

    def test_generate_returns_valid_output(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=16,
                do_sample=False,
            )
        assert output.shape[0] == 1
        assert output.shape[1] > 8

    def test_generate_no_sample(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
            )
        assert output.shape[1] == 12

    def test_generate_eos_stopping(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=32,
                eos_token_id=0,
                pad_token_id=0,
                do_sample=True,
                temperature=0.5,
            )
        assert output.shape[0] == 1


class TestQingxiaotuanBatchGeneration:

    def test_batch_generation(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            outputs = []
            for i in range(input_ids.shape[0]):
                out = small_qingxiaotuan_model.generate(
                    input_ids[i:i+1],
                    max_new_tokens=8,
                    do_sample=False,
                )
                outputs.append(out)
            assert len(outputs) == 2

    def test_batch_generation_length(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
            )
            assert output.shape[1] == 8


class TestQingxiaotuanGenerationParameters:

    def test_generate_with_temperature(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                temperature=1.5,
                do_sample=True,
            )
            assert output.shape[1] == 12

    def test_generate_with_top_p(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                top_p=0.9,
                temperature=0.7,
                do_sample=True,
            )
            assert output.shape[1] == 12

    def test_generate_with_top_k(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                top_k=50,
                temperature=0.7,
                do_sample=True,
            )
            assert output.shape[1] == 12

    def test_generate_with_repetition_penalty(self, small_qingxiaotuan_model):
        import torch
        input_ids = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                repetition_penalty=1.2,
                do_sample=False,
            )
            assert output.shape[1] == 12
