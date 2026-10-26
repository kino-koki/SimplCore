# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Java桥接层
支持Java通过HTTP协议调用SimplCore框架
"""
from __future__ import annotations

import json
import threading
import numpy as np
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler


class JavaBridgeHandler(BaseHTTPRequestHandler):
    """Java桥接HTTP请求处理器"""
    
    _model_registry: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    def _json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        """返回JSON响应"""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def _json_request(self) -> Dict[str, Any]:
        """解析JSON请求"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(body)
    
    def do_POST(self) -> None:
        """处理POST请求"""
        try:
            if self.path == "/api/tensor/create":
                self._handle_tensor_create()
            elif self.path == "/api/tensor/operation":
                self._handle_tensor_operation()
            elif self.path == "/api/model/create":
                self._handle_model_create()
            elif self.path == "/api/model/train":
                self._handle_model_train()
            elif self.path == "/api/model/infer":
                self._handle_model_infer()
            elif self.path == "/api/model/save":
                self._handle_model_save()
            elif self.path == "/api/model/load":
                self._handle_model_load()
            else:
                self._json_response(404, {"error": "Not found"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})
    
    def _handle_tensor_create(self) -> None:
        """创建张量"""
        from ..engine import randn, zeros, ones
        
        data = self._json_request()
        shape = tuple(data.get("shape", [2, 2]))
        dtype = data.get("dtype", "float32")
        init_type = data.get("init", "randn")
        
        if init_type == "randn":
            tensor = randn(*shape)
        elif init_type == "zeros":
            tensor = zeros(*shape)
        elif init_type == "ones":
            tensor = ones(*shape)
        else:
            tensor = randn(*shape)
        
        result = {
            "id": id(tensor),
            "shape": list(tensor.shape),
            "dtype": dtype,
            "requires_grad": tensor.requires_grad,
            "data": tensor._data.tolist(),
        }
        self._json_response(200, result)
    
    def _handle_tensor_operation(self) -> None:
        """张量操作"""
        from ..engine import SimpTensor
        
        data = self._json_request()
        op = data.get("op")
        tensor_data = data.get("tensor", [])
        args = data.get("args", {})
        
        tensor = SimpTensor(np.array(tensor_data))
        
        if op == "matmul":
            other_data = args.get("other", [])
            other = SimpTensor(np.array(other_data))
            result = tensor.matmul(other)
        elif op == "add":
            other = args.get("other", 0)
            result = tensor + other
        elif op == "mul":
            other = args.get("other", 1)
            result = tensor * other
        elif op == "relu":
            from ..engine import relu
            result = relu(tensor)
        elif op == "softmax":
            from ..engine import softmax
            result = softmax(tensor)
        elif op == "sum":
            result = tensor.sum()
        elif op == "mean":
            result = tensor.mean()
        elif op == "reshape":
            shape = tuple(args.get("shape", []))
            result = tensor.reshape(shape)
        elif op == "transpose":
            dims = tuple(args.get("dims", [0, 1]))
            result = tensor.transpose(*dims)
        else:
            self._json_response(400, {"error": f"Unknown operation: {op}"})
            return
        
        self._json_response(200, {
            "shape": list(result.shape),
            "data": result._data.tolist(),
        })
    
    def _handle_model_create(self) -> None:
        """创建模型"""
        from ..engine import Sequential, Linear, ReLU, Conv2d, LayerNorm
        
        data = self._json_request()
        model_name = data.get("name")
        layers = data.get("layers", [])
        
        module_layers = []
        for layer_spec in layers:
            layer_type = layer_spec.get("type")
            if layer_type == "Linear":
                module_layers.append(Linear(
                    in_features=layer_spec.get("in_features", 10),
                    out_features=layer_spec.get("out_features", 10),
                ))
            elif layer_type == "ReLU":
                module_layers.append(ReLU())
            elif layer_type == "Conv2d":
                module_layers.append(Conv2d(
                    in_channels=layer_spec.get("in_channels", 1),
                    out_channels=layer_spec.get("out_channels", 1),
                    kernel_size=layer_spec.get("kernel_size", 3),
                ))
            elif layer_type == "LayerNorm":
                module_layers.append(LayerNorm(
                    normalized_shape=layer_spec.get("normalized_shape", 10),
                ))
            elif layer_type == "Flatten":
                from ..engine import Flatten
                module_layers.append(Flatten())
        
        model = Sequential(*module_layers)
        
        with self._lock:
            self._model_registry[model_name] = model
        
        self._json_response(200, {"status": "success", "model_name": model_name})
    
    def _handle_model_train(self) -> None:
        """训练模型"""
        from ..engine import Adam, cross_entropy, randn
        
        data = self._json_request()
        model_name = data.get("model_name")
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 32)
        
        with self._lock:
            model = self._model_registry.get(model_name)
        
        if model is None:
            self._json_response(404, {"error": f"Model not found: {model_name}"})
            return
        
        optimizer = Adam(model.parameters(), lr=1e-3)
        
        losses = []
        for epoch in range(epochs):
            x = randn(batch_size, 10)
            y = randn(batch_size, 5)
            
            optimizer.zero_grad()
            output = model(x)
            loss = cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            
            losses.append(float(loss._data))
        
        self._json_response(200, {"losses": losses, "final_loss": losses[-1]})
    
    def _handle_model_infer(self) -> None:
        """模型推理"""
        from ..engine import SimpTensor
        
        data = self._json_request()
        model_name = data.get("model_name")
        input_data = data.get("input", [])
        
        with self._lock:
            model = self._model_registry.get(model_name)
        
        if model is None:
            self._json_response(404, {"error": f"Model not found: {model_name}"})
            return
        
        input_tensor = SimpTensor(np.array(input_data))
        output = model(input_tensor)
        
        self._json_response(200, {"output": output._data.tolist()})
    
    def _handle_model_save(self) -> None:
        """保存模型"""
        data = self._json_request()
        model_name = data.get("model_name")
        path = data.get("path", "./model.npz")
        
        with self._lock:
            model = self._model_registry.get(model_name)
        
        if model is None:
            self._json_response(404, {"error": f"Model not found: {model_name}"})
            return
        
        checkpoint = {}
        for name, param in model.named_parameters():
            checkpoint[f"param_{name}"] = param._data
        
        np.savez(path, **checkpoint)
        self._json_response(200, {"status": "success", "path": path})
    
    def _handle_model_load(self) -> None:
        """加载模型"""
        from ..engine import Sequential, Linear, ReLU
        
        data = self._json_request()
        model_name = data.get("model_name")
        path = data.get("path", "./model.npz")
        
        checkpoint = np.load(path)
        
        model = Sequential(
            Linear(10, 10),
            ReLU(),
            Linear(10, 5),
        )
        
        for name, param in model.named_parameters():
            key = f"param_{name}"
            if key in checkpoint:
                param._data = checkpoint[key]
        
        with self._lock:
            self._model_registry[model_name] = model
        
        self._json_response(200, {"status": "success", "model_name": model_name})


class JavaBridgeServer:
    """Java桥接服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动服务器"""
        self._server = HTTPServer((self.host, self.port), JavaBridgeHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"JavaBridgeServer started on http://{self.host}:{self.port}")
    
    def stop(self) -> None:
        """停止服务器"""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join()
    
    def is_running(self) -> bool:
        """检查服务器是否运行"""
        return self._server is not None


def start_java_bridge(host: str = "localhost", port: int = 8080) -> JavaBridgeServer:
    """启动Java桥接服务器"""
    server = JavaBridgeServer(host, port)
    server.start()
    return server


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore Java桥接层自检")
    print("=" * 60)
    
    server = start_java_bridge()
    
    import time
    import urllib.request
    import urllib.parse
    
    time.sleep(1)
    
    test_data = {
        "shape": [3, 4],
        "init": "randn",
    }
    
    req = urllib.request.Request(
        "http://localhost:8080/api/tensor/create",
        data=json.dumps(test_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"Tensor create result: {result['shape']}")
        print("JavaBridge 自检 PASSED")
    except Exception as e:
        print(f"JavaBridge 自检 FAILED: {e}")
    
    server.stop()


