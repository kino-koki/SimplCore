package com.qingxiaotuan.simpcore.cognition;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class NeuralMachineInterfaceProtocol {

    public enum ProtocolVersion {
        NMI_1_0("1.0"),
        NMI_1_1("1.1"),
        NMI_2_0("2.0");

        private final String version;

        ProtocolVersion(String version) {
            this.version = version;
        }

        public String getVersion() { return version; }
    }

    public enum MessageType {
        INIT,
        SYNC,
        DATA,
        CONTROL,
        STATUS,
        ERROR,
        ACK,
        NACK
    }

    public enum DataType {
        EEG,
        EMG,
        EOG,
        ECG,
        fMRI,
        BCI_COMMAND,
        FEEDBACK
    }

    public enum ControlCommand {
        START_STREAMING,
        STOP_STREAMING,
        CALIBRATE,
        RESET,
        CONFIGURE,
        QUERY_STATUS
    }

    private final ProtocolVersion version;
    private final ConcurrentHashMap<String, DataListener> listeners;
    private ServerSocket serverSocket;
    private Socket clientSocket;
    private volatile boolean connected = false;
    private ExecutorService executorService;
    private final Object lock = new Object();

    public NeuralMachineInterfaceProtocol() {
        this.version = ProtocolVersion.NMI_2_0;
        this.listeners = new ConcurrentHashMap<>();
        this.executorService = Executors.newFixedThreadPool(4);
    }

    public void startServer(int port) throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("NMI Protocol Server started on port " + port);
        
        executorService.submit(() -> {
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    Socket client = serverSocket.accept();
                    synchronized (lock) {
                        this.clientSocket = client;
                        this.connected = true;
                    }
                    System.out.println("Client connected: " + client.getInetAddress());
                    startMessageLoop(client);
                }
            } catch (IOException e) {
                if (!serverSocket.isClosed()) {
                    System.err.println("Server error: " + e.getMessage());
                }
            }
        });
    }

    public void connect(String host, int port) throws IOException {
        clientSocket = new Socket(host, port);
        connected = true;
        System.out.println("Connected to NMI server: " + host + ":" + port);
        startMessageLoop(clientSocket);
    }

    private void startMessageLoop(Socket socket) {
        executorService.submit(() -> {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {
                
                String line;
                while ((line = in.readLine()) != null) {
                    processMessage(line, out);
                }
                
                synchronized (lock) {
                    connected = false;
                    clientSocket = null;
                }
                System.out.println("Client disconnected");
                
            } catch (IOException e) {
                System.err.println("Message loop error: " + e.getMessage());
                synchronized (lock) {
                    connected = false;
                    clientSocket = null;
                }
            }
        });
    }

    private void processMessage(String message, PrintWriter out) {
        try {
            NMIMessage nmiMessage = deserializeMessage(message);
            
            switch (nmiMessage.getMessageType()) {
                case INIT:
                    handleInit(nmiMessage, out);
                    break;
                case SYNC:
                    handleSync(nmiMessage, out);
                    break;
                case DATA:
                    handleData(nmiMessage);
                    break;
                case CONTROL:
                    handleControl(nmiMessage, out);
                    break;
                case STATUS:
                    handleStatus(nmiMessage, out);
                    break;
                case ERROR:
                    handleError(nmiMessage);
                    break;
                case ACK:
                    handleAck(nmiMessage);
                    break;
                case NACK:
                    handleNack(nmiMessage);
                    break;
            }
        } catch (Exception e) {
            System.err.println("Failed to process message: " + e.getMessage());
            sendMessage(new NMIMessage(MessageType.ERROR, "Invalid message format"));
        }
    }

    private void handleInit(NMIMessage message, PrintWriter out) {
        System.out.println("Init received: " + message.getPayload());
        sendAck(out, "Initialization complete");
    }

    private void handleSync(NMIMessage message, PrintWriter out) {
        long timestamp = System.currentTimeMillis();
        sendMessage(new NMIMessage(MessageType.SYNC, String.valueOf(timestamp)));
    }

    private void handleData(NMIMessage message) {
        String dataType = message.getHeaders().get("dataType");
        if (dataType != null) {
            DataListener listener = listeners.get(dataType);
            if (listener != null) {
                listener.onDataReceived(message.getPayload());
            }
        }
    }

    private void handleControl(NMIMessage message, PrintWriter out) {
        String command = message.getPayload();
        System.out.println("Control command: " + command);
        
        try {
            ControlCommand cmd = ControlCommand.valueOf(command);
            switch (cmd) {
                case START_STREAMING:
                    startStreaming();
                    sendAck(out, "Streaming started");
                    break;
                case STOP_STREAMING:
                    stopStreaming();
                    sendAck(out, "Streaming stopped");
                    break;
                case CALIBRATE:
                    calibrate();
                    sendAck(out, "Calibration complete");
                    break;
                case RESET:
                    reset();
                    sendAck(out, "System reset");
                    break;
                case CONFIGURE:
                    configure(message.getHeaders());
                    sendAck(out, "Configuration applied");
                    break;
                case QUERY_STATUS:
                    sendStatus(out);
                    break;
            }
        } catch (IllegalArgumentException e) {
            sendNack(out, "Unknown command: " + command);
        }
    }

    private void handleStatus(NMIMessage message, PrintWriter out) {
        sendStatus(out);
    }

    private void handleError(NMIMessage message) {
        System.err.println("NMI Error: " + message.getPayload());
    }

    private void handleAck(NMIMessage message) {
        System.out.println("ACK: " + message.getPayload());
    }

    private void handleNack(NMIMessage message) {
        System.err.println("NACK: " + message.getPayload());
    }

    private void sendAck(PrintWriter out, String message) {
        out.println(serializeMessage(new NMIMessage(MessageType.ACK, message)));
    }

    private void sendNack(PrintWriter out, String message) {
        out.println(serializeMessage(new NMIMessage(MessageType.NACK, message)));
    }

    private void sendStatus(PrintWriter out) {
        Map<String, String> headers = new HashMap<>();
        headers.put("connected", String.valueOf(connected));
        headers.put("version", version.getVersion());
        headers.put("timestamp", String.valueOf(System.currentTimeMillis()));
        NMIMessage status = new NMIMessage(MessageType.STATUS, headers, "OK");
        out.println(serializeMessage(status));
    }

    public void sendData(DataType dataType, String data) {
        if (!connected) {
            System.err.println("Not connected");
            return;
        }
        
        Map<String, String> headers = new HashMap<>();
        headers.put("dataType", dataType.name());
        headers.put("timestamp", String.valueOf(System.currentTimeMillis()));
        headers.put("length", String.valueOf(data.length()));
        
        NMIMessage message = new NMIMessage(MessageType.DATA, headers, data);
        sendMessage(message);
    }

    public void sendControl(ControlCommand command) {
        sendMessage(new NMIMessage(MessageType.CONTROL, command.name()));
    }

    public void sendMessage(NMIMessage message) {
        synchronized (lock) {
            if (clientSocket != null && connected) {
                try {
                    PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                    out.println(serializeMessage(message));
                } catch (IOException e) {
                    System.err.println("Failed to send message: " + e.getMessage());
                    connected = false;
                }
            }
        }
    }

    private String serializeMessage(NMIMessage message) {
        StringBuilder sb = new StringBuilder();
        sb.append("NMI/").append(version.getVersion()).append(" ");
        sb.append(message.getMessageType().name()).append("\r\n");
        
        for (Map.Entry<String, String> entry : message.getHeaders().entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\r\n");
        }
        
        sb.append("\r\n");
        sb.append(message.getPayload());
        
        return sb.toString();
    }

    private NMIMessage deserializeMessage(String raw) {
        String[] lines = raw.split("\r\n");
        
        String[] firstLine = lines[0].split(" ");
        String versionStr = firstLine[0].substring(4);
        MessageType type = MessageType.valueOf(firstLine[1]);
        
        Map<String, String> headers = new HashMap<>();
        int i = 1;
        while (i < lines.length && !lines[i].isEmpty()) {
            String[] header = lines[i].split(": ", 2);
            if (header.length == 2) {
                headers.put(header[0], header[1]);
            }
            i++;
        }
        
        StringBuilder payload = new StringBuilder();
        for (int j = i + 1; j < lines.length; j++) {
            if (j > i + 1) payload.append("\n");
            payload.append(lines[j]);
        }
        
        return new NMIMessage(type, headers, payload.toString());
    }

    public void addDataListener(DataType dataType, DataListener listener) {
        listeners.put(dataType.name(), listener);
    }

    public void removeDataListener(DataType dataType) {
        listeners.remove(dataType.name());
    }

    private void startStreaming() {
        System.out.println("Starting data streaming...");
    }

    private void stopStreaming() {
        System.out.println("Stopping data streaming...");
    }

    private void calibrate() {
        System.out.println("Calibrating...");
    }

    private void reset() {
        System.out.println("Resetting system...");
    }

    private void configure(Map<String, String> config) {
        System.out.println("Configuring with: " + config);
    }

    public boolean isConnected() {
        return connected;
    }

    public void disconnect() {
        synchronized (lock) {
            connected = false;
            if (clientSocket != null) {
                try {
                    clientSocket.close();
                } catch (IOException e) {
                    // Ignore
                }
                clientSocket = null;
            }
            if (serverSocket != null) {
                try {
                    serverSocket.close();
                } catch (IOException e) {
                    // Ignore
                }
            }
        }
        executorService.shutdown();
    }

    public interface DataListener {
        void onDataReceived(String data);
    }

    public static class NMIMessage {
        private final MessageType messageType;
        private final Map<String, String> headers;
        private final String payload;

        public NMIMessage(MessageType messageType, String payload) {
            this(messageType, new HashMap<>(), payload);
        }

        public NMIMessage(MessageType messageType, Map<String, String> headers, String payload) {
            this.messageType = messageType;
            this.headers = headers != null ? headers : new HashMap<>();
            this.payload = payload != null ? payload : "";
        }

        public MessageType getMessageType() { return messageType; }
        public Map<String, String> getHeaders() { return headers; }
        public String getPayload() { return payload; }
    }

    public static class EEGData {
        private final double[] channels;
        private final long timestamp;
        private final float sampleRate;
        private final int channelCount;

        public EEGData(double[] channels, long timestamp, float sampleRate) {
            this.channels = channels;
            this.timestamp = timestamp;
            this.sampleRate = sampleRate;
            this.channelCount = channels.length;
        }

        public String toJSON() {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"type\": \"EEG\",");
            sb.append("\"timestamp\": ").append(timestamp).append(",");
            sb.append("\"sampleRate\": ").append(sampleRate).append(",");
            sb.append("\"channelCount\": ").append(channelCount).append(",");
            sb.append("\"data\": [");
            for (int i = 0; i < channels.length; i++) {
                sb.append(String.format("%.6f", channels[i]));
                if (i < channels.length - 1) sb.append(",");
            }
            sb.append("]");
            sb.append("}");
            return sb.toString();
        }

        public static EEGData fromJSON(String json) {
            int start = json.indexOf("[");
            int end = json.lastIndexOf("]");
            if (start == -1 || end == -1) return null;
            
            String dataStr = json.substring(start + 1, end);
            String[] parts = dataStr.split(",");
            double[] channels = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                channels[i] = Double.parseDouble(parts[i].trim());
            }
            
            long timestamp = extractLong(json, "timestamp");
            float sampleRate = extractFloat(json, "sampleRate");
            
            return new EEGData(channels, timestamp, sampleRate);
        }

        private static long extractLong(String json, String key) {
            int start = json.indexOf("\"" + key + "\":");
            if (start == -1) return 0;
            start = json.indexOf(":", start + key.length() + 3);
            int end = json.indexOf(",", start + 1);
            if (end == -1) end = json.indexOf("}", start + 1);
            return Long.parseLong(json.substring(start + 1, end).trim());
        }

        private static float extractFloat(String json, String key) {
            int start = json.indexOf("\"" + key + "\":");
            if (start == -1) return 0;
            start = json.indexOf(":", start + key.length() + 3);
            int end = json.indexOf(",", start + 1);
            if (end == -1) end = json.indexOf("}", start + 1);
            return Float.parseFloat(json.substring(start + 1, end).trim());
        }

        public double[] getChannels() { return channels; }
        public long getTimestamp() { return timestamp; }
        public float getSampleRate() { return sampleRate; }
        public int getChannelCount() { return channelCount; }
    }
}