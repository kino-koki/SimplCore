package com.qingxiaotuan.simpcore.api;

import java.util.*;
import java.util.concurrent.*;
import java.io.*;
import java.net.*;

public class APIServer {
    
    public interface RequestHandler {
        String handle(String path, String method, Map<String, String> params);
    }
    
    public interface WebSocketCallback {
        void onMessage(String message);
        void onClose();
    }
    
    private final Map<String, RequestHandler> handlers;
    private final List<WebSocketCallback> webSocketCallbacks;
    private final ExecutorService executor;
    private ServerSocket serverSocket;
    private volatile boolean running;
    private final int port;
    private volatile int requestCount;
    private volatile long totalLatencyMs;
    
    public APIServer(int port) {
        this.port = port;
        this.handlers = new ConcurrentHashMap<>();
        this.webSocketCallbacks = Collections.synchronizedList(new ArrayList<>());
        this.executor = Executors.newFixedThreadPool(16);
        this.running = false;
        this.requestCount = 0;
        this.totalLatencyMs = 0;
        registerDefaultHandlers();
    }
    
    public APIServer() {
        this(8080);
    }
    
    private void registerDefaultHandlers() {
        handlers.put("/api/v1/generate", this::handleGenerate);
        handlers.put("/api/v1/chat", this::handleChat);
        handlers.put("/api/v1/models", this::handleModels);
        handlers.put("/api/v1/health", this::handleHealth);
        handlers.put("/api/v1/embeddings", this::handleEmbeddings);
        handlers.put("/api/v1/completions", this::handleCompletions);
        handlers.put("/api/v1/knowledge", this::handleKnowledge);
        handlers.put("/api/v1/agent", this::handleAgent);
    }
    
    public void registerHandler(String path, RequestHandler handler) {
        handlers.put(path.toLowerCase(), handler);
    }
    
    public void start() throws IOException {
        if (running) return;
        
        serverSocket = new ServerSocket(port);
        running = true;
        
        executor.submit(() -> {
            while (running) {
                try {
                    Socket clientSocket = serverSocket.accept();
                    executor.submit(() -> handleClient(clientSocket));
                } catch (IOException e) {
                    if (!running) break;
                }
            }
        });
        
        System.out.println("[APIServer] Started on port " + port);
    }
    
    public void stop() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (IOException e) {
            System.err.println("[APIServer] Stop error: " + e.getMessage());
        }
        executor.shutdown();
        System.out.println("[APIServer] Stopped");
    }
    
    private void handleClient(Socket clientSocket) {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
             PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {
            
            String line = in.readLine();
            if (line == null) return;
            
            long startTime = System.currentTimeMillis();
            String response = processRequest(line);
            long latency = System.currentTimeMillis() - startTime;
            
            synchronized (this) {
                requestCount++;
                totalLatencyMs += latency;
            }
            
            out.println("HTTP/1.1 200 OK");
            out.println("Content-Type: application/json");
            out.println("Access-Control-Allow-Origin: *");
            out.println();
            out.println(response);
            
        } catch (IOException e) {
            System.err.println("[APIServer] Client error: " + e.getMessage());
        }
    }
    
    private String processRequest(String requestLine) {
        String[] parts = requestLine.split(" ");
        if (parts.length < 2) return "{\"error\": \"Invalid request\"}";
        
        String method = parts[0];
        String path = parts[1];
        
        Map<String, String> params = new HashMap<>();
        if (path.contains("?")) {
            String[] pathParts = path.split("\\?", 2);
            path = pathParts[0];
            String query = pathParts[1];
            for (String param : query.split("&")) {
                String[] keyValue = param.split("=", 2);
                if (keyValue.length == 2) {
                    params.put(keyValue[0], URLDecoder.decode(keyValue[1]));
                }
            }
        }
        
        RequestHandler handler = handlers.get(path.toLowerCase());
        if (handler != null) {
            return handler.handle(path, method, params);
        }
        
        return "{\"error\": \"Not found\", \"path\": \"" + path + "\"}";
    }
    
    private String handleGenerate(String path, String method, Map<String, String> params) {
        String prompt = params.getOrDefault("prompt", "");
        int maxTokens = Integer.parseInt(params.getOrDefault("maxTokens", "100"));
        return String.format("{\"text\": \"Generated response for: %s\", \"maxTokens\": %d}", prompt, maxTokens);
    }
    
    private String handleChat(String path, String method, Map<String, String> params) {
        String message = params.getOrDefault("message", "");
        String history = params.getOrDefault("history", "[]");
        return String.format("{\"response\": \"Chat response to: %s\", \"history\": %s}", message, history);
    }
    
    private String handleModels(String path, String method, Map<String, String> params) {
        return "[{\"name\": \"SC-Tiny-1M\", \"params\": 1000000}, " +
               "{\"name\": \"SC-Small-10M\", \"params\": 10000000}, " +
               "{\"name\": \"SC-Medium-1B\", \"params\": 1000000000}, " +
               "{\"name\": \"SC-Large-7B\", \"params\": 7000000000}, " +
               "{\"name\": \"SC-Giant-70B\", \"params\": 70000000000}]";
    }
    
    private String handleHealth(String path, String method, Map<String, String> params) {
        return String.format("{\"status\": \"healthy\", \"requests\": %d, \"avgLatency\": %.2fms}",
            requestCount, getAvgLatency());
    }
    
    private String handleEmbeddings(String path, String method, Map<String, String> params) {
        String text = params.getOrDefault("text", "");
        int dimensions = Integer.parseInt(params.getOrDefault("dimensions", "768"));
        return String.format("{\"embedding\": [%d dimensions], \"text\": \"%s\"}", dimensions, text);
    }
    
    private String handleCompletions(String path, String method, Map<String, String> params) {
        String prompt = params.getOrDefault("prompt", "");
        int maxTokens = Integer.parseInt(params.getOrDefault("maxTokens", "50"));
        return String.format("{\"choices\": [{\"text\": \"Completion for: %s\"}], \"usage\": {\"total_tokens\": %d}}", prompt, maxTokens);
    }
    
    private String handleKnowledge(String path, String method, Map<String, String> params) {
        String query = params.getOrDefault("query", "");
        return String.format("{\"results\": [{\"content\": \"Knowledge result for: %s\", \"relevance\": 0.95}]}", query);
    }
    
    private String handleAgent(String path, String method, Map<String, String> params) {
        String input = params.getOrDefault("input", "");
        return String.format("{\"response\": \"Agent processed: %s\"}", input);
    }
    
    public void addWebSocketCallback(WebSocketCallback callback) {
        webSocketCallbacks.add(callback);
    }
    
    public void broadcastMessage(String message) {
        for (WebSocketCallback callback : webSocketCallbacks) {
            callback.onMessage(message);
        }
    }
    
    public synchronized int getRequestCount() {
        return requestCount;
    }
    
    public synchronized double getAvgLatency() {
        return requestCount > 0 ? (double) totalLatencyMs / requestCount : 0;
    }
    
    public boolean isRunning() {
        return running;
    }
    
    public int getPort() {
        return port;
    }
    
    public String getStatus() {
        return String.format(
            "APIServer: running=%s, port=%d, handlers=%d, requests=%d, avgLatency=%.2fms",
            running, port, handlers.size(), requestCount, getAvgLatency()
        );
    }
}