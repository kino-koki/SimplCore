package com.qingxiaotuan.simpcore.deployment;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class ModelDeployer {

    private final SimpCoreModel model;
    private final Tokenizer tokenizer;
    private ServerSocket serverSocket;
    private ExecutorService executorService;
    private volatile boolean running = false;
    private final ConcurrentHashMap<String, String> config = new ConcurrentHashMap<>();

    public static final String FORMAT_SIMPCORE = "simpcore";
    public static final String FORMAT_ONNX = "onnx";
    public static final String FORMAT_JSON = "json";

    public ModelDeployer(SimpCoreModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        initDefaultConfig();
    }

    private void initDefaultConfig() {
        config.put("host", "localhost");
        config.put("port", "8080");
        config.put("timeout", "30000");
        config.put("maxConnections", "100");
        config.put("enableCORS", "true");
        config.put("apiPrefix", "/api");
    }

    public void setConfig(String key, String value) {
        config.put(key, value);
    }

    public String getConfig(String key) {
        return config.get(key);
    }

    public void exportModel(String filePath, String format) throws IOException {
        switch (format.toLowerCase()) {
            case FORMAT_SIMPCORE:
                exportSimpCoreFormat(filePath);
                break;
            case FORMAT_ONNX:
                exportONNXFormat(filePath);
                break;
            case FORMAT_JSON:
                exportJSONFormat(filePath);
                break;
            default:
                throw new IllegalArgumentException("Unsupported format: " + format);
        }
        System.out.println("Model exported to: " + filePath);
    }

    private void exportSimpCoreFormat(String filePath) throws IOException {
        model.save(filePath);
    }

    private void exportONNXFormat(String filePath) throws IOException {
        StringBuilder onnx = new StringBuilder();
        onnx.append("ir_version: 8\n");
        onnx.append("opset_import: { version: 13 }\n");
        onnx.append("\n");
        onnx.append("graph {\n");
        onnx.append("  name: \"SimpCoreModel\"\n");
        onnx.append("\n");
        onnx.append("  input {\n");
        onnx.append("    name: \"input_ids\"\n");
        onnx.append("    type {\n");
        onnx.append("      tensor_type {\n");
        onnx.append("        elem_type: INT64\n");
        onnx.append("        shape: { dim: { dim_param: \"batch_size\" } }\n");
        onnx.append("        shape: { dim: { dim_param: \"seq_len\" } }\n");
        onnx.append("      }\n");
        onnx.append("    }\n");
        onnx.append("  }\n");
        onnx.append("\n");
        onnx.append("  output {\n");
        onnx.append("    name: \"logits\"\n");
        onnx.append("    type {\n");
        onnx.append("      tensor_type {\n");
        onnx.append("        elem_type: FLOAT\n");
        onnx.append("        shape: { dim: { dim_param: \"batch_size\" } }\n");
        onnx.append("        shape: { dim: { dim_param: \"seq_len\" } }\n");
        onnx.append("        shape: { dim: " + model.getVocabSize() + " }\n");
        onnx.append("      }\n");
        onnx.append("    }\n");
        onnx.append("  }\n");
        onnx.append("\n");
        onnx.append("  node {\n");
        onnx.append("    name: \"SimpCoreInference\"\n");
        onnx.append("    op_type: \"Custom\"\n");
        onnx.append("    input: \"input_ids\"\n");
        onnx.append("    output: \"logits\"\n");
        onnx.append("    attribute {\n");
        onnx.append("      name: \"model_type\"\n");
        onnx.append("      s: \"SimpCore\"\n");
        onnx.append("      type: STRING\n");
        onnx.append("    }\n");
        onnx.append("    attribute {\n");
        onnx.append("      name: \"d_model\"\n");
        onnx.append("      i: " + model.getDModel() + "\n");
        onnx.append("      type: INT\n");
        onnx.append("    }\n");
        onnx.append("    attribute {\n");
        onnx.append("      name: \"n_layers\"\n");
        onnx.append("      i: " + model.getNLayers() + "\n");
        onnx.append("      type: INT\n");
        onnx.append("    }\n");
        onnx.append("    attribute {\n");
        onnx.append("      name: \"vocab_size\"\n");
        onnx.append("      i: " + model.getVocabSize() + "\n");
        onnx.append("      type: INT\n");
        onnx.append("    }\n");
        onnx.append("  }\n");
        onnx.append("}\n");

        try (PrintWriter writer = new PrintWriter(filePath)) {
            writer.write(onnx.toString());
        }
    }

    private void exportJSONFormat(String filePath) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"model_type\": \"SimpCore\",\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"vocab_size\": " + model.getVocabSize() + ",\n");
        json.append("    \"d_model\": " + model.getDModel() + ",\n");
        json.append("    \"n_layers\": " + model.getNLayers() + ",\n");
        json.append("    \"max_seq_len\": " + model.getMaxSeqLen() + ",\n");
        json.append("    \"total_parameters\": " + model.getTotalParameters() + ",\n");
        json.append("    \"export_time\": \"" + new Date().toString() + "\"\n");
        json.append("  },\n");
        json.append("  \"architecture\": {\n");
        json.append("    \"embedding\": {\n");
        json.append("      \"input_dim\": " + model.getVocabSize() + ",\n");
        json.append("      \"output_dim\": " + model.getDModel() + "\n");
        json.append("    },\n");
        json.append("    \"layers\": [\n");

        for (int l = 0; l < model.getNLayers(); l++) {
            json.append("      {\n");
            json.append("        \"layer_type\": \"SimpCoreLayer\",\n");
            json.append("        \"index\": " + l + ",\n");
            json.append("        \"components\": [\n");
            json.append("          \"AdaptiveBasisLayer\",\n");
            json.append("          \"DynamicMemoryChannel\",\n");
            json.append("          \"TopologyAwareRouter\",\n");
            json.append("          \"CausalInterventionNetwork\",\n");
            json.append("          \"InformationBottleneck\",\n");
            json.append("          \"FeedForwardNetwork\",\n");
            json.append("          \"MetaRepresentationDistiller\",\n");
            json.append("          \"PlasticityUpdateModule\"\n");
            json.append("        ]\n");
            json.append("      }" + (l < model.getNLayers() - 1 ? "," : "") + "\n");
        }

        json.append("    ],\n");
        json.append("    \"output\": {\n");
        json.append("      \"input_dim\": " + model.getDModel() + ",\n");
        json.append("      \"output_dim\": " + model.getVocabSize() + "\n");
        json.append("    }\n");
        json.append("  },\n");
        json.append("  \"training_config\": {\n");
        json.append("    \"learning_rate\": 0.001,\n");
        json.append("    \"gradient_clipping\": 1.0\n");
        json.append("  }\n");
        json.append("}\n");

        try (PrintWriter writer = new PrintWriter(filePath)) {
            writer.write(json.toString());
        }
    }

    public void startServer() throws IOException {
        String host = config.get("host");
        int port = Integer.parseInt(config.get("port"));
        int maxConnections = Integer.parseInt(config.get("maxConnections"));

        serverSocket = new ServerSocket(port, maxConnections, InetAddress.getByName(host));
        executorService = Executors.newFixedThreadPool(maxConnections);
        running = true;

        System.out.println("SimpCore Server started on " + host + ":" + port);
        System.out.println("API endpoints:");
        System.out.println("  POST " + config.get("apiPrefix") + "/generate");
        System.out.println("  POST " + config.get("apiPrefix") + "/predict");
        System.out.println("  GET " + config.get("apiPrefix") + "/health");
        System.out.println("  GET " + config.get("apiPrefix") + "/info");

        new Thread(() -> {
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    executorService.submit(new RequestHandler(client));
                } catch (IOException e) {
                    if (!running) break;
                    System.err.println("Error accepting connection: " + e.getMessage());
                }
            }
        }).start();
    }

    public void stopServer() {
        running = false;
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
                System.err.println("Error closing server: " + e.getMessage());
            }
        }
        if (executorService != null) {
            executorService.shutdown();
        }
        System.out.println("SimpCore Server stopped");
    }

    private class RequestHandler implements Runnable {
        private final Socket client;

        public RequestHandler(Socket client) {
            this.client = client;
        }

        @Override
        public void run() {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
                 PrintWriter out = new PrintWriter(client.getOutputStream(), true)) {

                String line = in.readLine();
                if (line == null) return;

                String[] parts = line.split(" ");
                if (parts.length < 2) {
                    sendError(out, 400, "Bad Request");
                    return;
                }

                String method = parts[0];
                String path = parts[1];

                if (config.get("enableCORS").equals("true")) {
                    out.println("Access-Control-Allow-Origin: *");
                    out.println("Access-Control-Allow-Methods: GET, POST, OPTIONS");
                    out.println("Access-Control-Allow-Headers: Content-Type");
                }

                if (method.equals("OPTIONS")) {
                    out.println("HTTP/1.1 200 OK");
                    out.println();
                    return;
                }

                switch (path) {
                    case "/api/generate":
                        handleGenerate(in, out);
                        break;
                    case "/api/predict":
                        handlePredict(in, out);
                        break;
                    case "/api/health":
                        handleHealth(out);
                        break;
                    case "/api/info":
                        handleInfo(out);
                        break;
                    default:
                        sendError(out, 404, "Not Found");
                }

            } catch (IOException e) {
                System.err.println("Error handling request: " + e.getMessage());
            } finally {
                try {
                    client.close();
                } catch (IOException e) {
                    // Ignore
                }
            }
        }

        private void handleGenerate(BufferedReader in, PrintWriter out) throws IOException {
            StringBuilder body = new StringBuilder();
            String line;
            while ((line = in.readLine()) != null && !line.isEmpty()) {
                body.append(line);
            }

            String json = body.toString();
            String prompt = extractField(json, "prompt");
            int maxTokens = extractIntField(json, "max_tokens", 50);
            int eosTokenId = extractIntField(json, "eos_token_id", tokenizer.getEosTokenId());

            String result = model.generateText(prompt, tokenizer, maxTokens, eosTokenId);

            out.println("HTTP/1.1 200 OK");
            out.println("Content-Type: application/json");
            out.println();
            out.println("{\"generated_text\": \"" + escapeJson(result) + "\"}");
        }

        private void handlePredict(BufferedReader in, PrintWriter out) throws IOException {
            StringBuilder body = new StringBuilder();
            String line;
            while ((line = in.readLine()) != null && !line.isEmpty()) {
                body.append(line);
            }

            String json = body.toString();
            String text = extractField(json, "text");

            int[] inputIds = tokenizer.encode(text);
            float[] logits = model.forward(inputIds, 1, inputIds.length);

            StringBuilder logitsJson = new StringBuilder("[");
            for (int i = 0; i < Math.min(10, logits.length); i++) {
                logitsJson.append(logits[i]);
                if (i < Math.min(10, logits.length) - 1) logitsJson.append(",");
            }
            logitsJson.append("]");

            out.println("HTTP/1.1 200 OK");
            out.println("Content-Type: application/json");
            out.println();
            out.println("{\"logits\": " + logitsJson + ", \"length\": " + logits.length + "}");
        }

        private void handleHealth(PrintWriter out) {
            out.println("HTTP/1.1 200 OK");
            out.println("Content-Type: application/json");
            out.println();
            out.println("{\"status\": \"healthy\", \"model_loaded\": true}");
        }

        private void handleInfo(PrintWriter out) {
            out.println("HTTP/1.1 200 OK");
            out.println("Content-Type: application/json");
            out.println();
            out.println("{\n" +
                    "  \"model_type\": \"SimpCore\",\n" +
                    "  \"vocab_size\": " + model.getVocabSize() + ",\n" +
                    "  \"d_model\": " + model.getDModel() + ",\n" +
                    "  \"n_layers\": " + model.getNLayers() + ",\n" +
                    "  \"max_seq_len\": " + model.getMaxSeqLen() + ",\n" +
                    "  \"total_parameters\": " + model.getTotalParameters() + "\n" +
                    "}");
        }

        private void sendError(PrintWriter out, int code, String message) {
            out.println("HTTP/1.1 " + code + " " + message);
            out.println("Content-Type: application/json");
            out.println();
            out.println("{\"error\": \"" + message + "\"}");
        }

        private String extractField(String json, String field) {
            int start = json.indexOf("\"" + field + "\":");
            if (start == -1) return "";
            start = json.indexOf("\"", start + field.length() + 3);
            if (start == -1) return "";
            int end = json.indexOf("\"", start + 1);
            if (end == -1) return "";
            return json.substring(start + 1, end);
        }

        private int extractIntField(String json, String field, int defaultValue) {
            int start = json.indexOf("\"" + field + "\":");
            if (start == -1) return defaultValue;
            start = json.indexOf(":", start + field.length() + 3);
            if (start == -1) return defaultValue;
            int end = json.indexOf(",", start + 1);
            if (end == -1) end = json.indexOf("}", start + 1);
            if (end == -1) return defaultValue;
            try {
                return Integer.parseInt(json.substring(start + 1, end).trim());
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }

        private String escapeJson(String str) {
            return str.replace("\\", "\\\\")
                      .replace("\"", "\\\"")
                      .replace("\n", "\\n")
                      .replace("\r", "\\r");
        }
    }

    public static SimpCoreModel loadModel(String filePath) throws IOException {
        return SimpCoreModel.load(filePath);
    }

    public static ModelDeployer createAndStart(SimpCoreModel model, Tokenizer tokenizer, 
                                               String host, int port) throws IOException {
        ModelDeployer deployer = new ModelDeployer(model, tokenizer);
        deployer.setConfig("host", host);
        deployer.setConfig("port", String.valueOf(port));
        deployer.startServer();
        return deployer;
    }
}