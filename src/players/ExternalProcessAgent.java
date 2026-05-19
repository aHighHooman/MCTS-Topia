package players;

import core.actions.Action;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import players.external.ExternalBotPayloadBuilder;
import players.external.ExternalForwardModelSession;
import utils.ElapsedCpuTimer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class ExternalProcessAgent extends Agent {

    private final List<String> command;
    private final boolean forwardModelEnabled;
    private final Options options;
    private transient Process process;
    private transient BufferedWriter processIn;
    private transient BufferedReader processOut;
    private transient Thread stderrPump;
    private transient BufferedWriter stdoutLogWriter;
    private transient BufferedWriter stderrLogWriter;
    private transient long matchStartMillis;
    private int requestCounter;

    public ExternalProcessAgent(long seed, List<String> command) {
        this(seed, command, new Options());
    }

    public ExternalProcessAgent(long seed, List<String> command, Options options) {
        super(seed);
        this.command = new ArrayList<>(command);
        this.options = options == null ? new Options() : options.copy();
        this.forwardModelEnabled = !disablesForwardModel(command);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        ArrayList<Action> legalActions = gs.getAllAvailableActions();
        if (legalActions.isEmpty()) {
            return null;
        }

        ensureProcessStarted();
        ensureMatchWithinLifetime();

        ArrayList<String> actionIds = new ArrayList<>(legalActions.size());
        for (int i = 0; i < legalActions.size(); i++) {
            actionIds.add("A" + i);
        }

        ExternalForwardModelSession forwardModelSession = forwardModelEnabled
                ? new ExternalForwardModelSession(gs, playerID, legalActions, actionIds)
                : null;

        JSONObject request = ExternalBotPayloadBuilder.buildActionRequest(
                gs,
                playerID,
                Math.max(0L, ect.remainingTimeMillis()),
                requestCounter++,
                legalActions,
                actionIds,
                forwardModelSession != null ? forwardModelSession.getRootStateId() : null,
                forwardModelEnabled
        );

        try {
            writeRequest(request);
            long deadline = computeDeadlineMillis(ect);

            while (true) {
                String line = readResponseLine(deadline);

                JSONObject response = new JSONObject(line);
                String messageType = response.optString("type", "");
                if ("forward_model".equals(messageType) && forwardModelSession != null) {
                    JSONObject fmResponse = forwardModelSession.handleCommand(response);
                    writeRequest(fmResponse);
                    continue;
                }

                JSONObject selectedAction = response.optJSONObject("action");
                JSONObject actionResponse = selectedAction == null ? response : selectedAction;
                String actionId = actionResponse.optString("actionId", actionResponse.optString("action_id", null));
                int actionIndex = compactActionIndex(actionResponse);
                if (actionIndex >= 0 && actionIndex < legalActions.size()) {
                    return legalActions.get(actionIndex);
                }

                if (actionId != null) {
                    int selectedIdx = actionIds.indexOf(actionId);
                    if (selectedIdx >= 0) {
                        return legalActions.get(selectedIdx);
                    }
                }

                Action fallbackAction = fallbackActionFromRanking(request, response, legalActions, actionIds);
                if (fallbackAction != null) {
                    return fallbackAction;
                }

                throw new IllegalStateException("External bot returned an invalid action for player " + playerID +
                        ": " + response);
            }
        } catch (Exception e) {
            destroyProcess();
            throw new RuntimeException("Failed to communicate with external bot for player " + playerID +
                    " using command " + command + ": " + e.getMessage(), e);
        }
    }

    private int compactActionIndex(JSONObject response) {
        if (response.has("i")) {
            return response.optInt("i", -1);
        }
        if (response.has("action_index")) {
            return response.optInt("action_index", -1);
        }
        if (response.has("actionIndex")) {
            return response.optInt("actionIndex", -1);
        }
        return -1;
    }

    @Override
    public void result(GameState gs, double reward) {
        boolean sentResult = false;
        if (processIn != null) {
            try {
                JSONObject resultPayload = ExternalBotPayloadBuilder.buildResult(gs, playerID, reward);
                processIn.write(resultPayload.toString());
                processIn.newLine();
                processIn.flush();
                sentResult = true;
            } catch (IOException e) {
                System.err.println("[ExternalProcessAgent] Failed to send result payload to bot for player " +
                        playerID + ": " + e.getMessage());
            }
        }
        if (sentResult) {
            awaitProcessExitAfterResult();
        }
        destroyProcess();
    }

    @Override
    public Agent copy() {
        return new ExternalProcessAgent(seed, command, options);
    }

    private void ensureProcessStarted() {
        if (process != null && process.isAlive()) {
            return;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.directory(options.workingDirectory);
            pb.environment().putIfAbsent("CUDA_MODULE_LOADING", "LAZY");
            pb.environment().putIfAbsent("OMP_NUM_THREADS", "1");
            pb.environment().putIfAbsent("MKL_NUM_THREADS", "1");
            pb.environment().putAll(options.environment);
            openLogs();
            process = pb.start();
            processIn = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            processOut = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            matchStartMillis = System.currentTimeMillis();
            logStdout("# command: " + String.join(" ", command));
            stderrPump = new Thread(() -> pumpStderr(process.getErrorStream()), "external-bot-stderr-" + playerID);
            stderrPump.setDaemon(true);
            stderrPump.start();
            if (!process.isAlive()) {
                throw new IOException("Process exited immediately with code " + process.exitValue());
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to start external bot process for player " + playerID +
                    " with command " + command, e);
        }
    }

    private void pumpStderr(InputStream errorStream) {
        try (BufferedReader err = new BufferedReader(new InputStreamReader(errorStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = err.readLine()) != null) {
                logStderr(line);
                System.err.println("[ExternalBot p" + playerID + "] " + line);
            }
        } catch (IOException e) {
            System.err.println("[ExternalProcessAgent] Error while reading bot stderr for player " + playerID +
                    ": " + e.getMessage());
        }
    }

    private void writeRequest(JSONObject request) throws IOException {
        String payload = request.toString();
        processIn.write(payload);
        processIn.newLine();
        processIn.flush();
        logStdout("> " + payload);
    }

    private String readResponseLine(long deadlineMillis) throws Exception {
        while (true) {
            if (processOut.ready()) {
                String line = processOut.readLine();
                if (line == null) {
                    throw new EOFException("Bot process ended unexpectedly for player " + playerID + ".");
                }
                logStdout("< " + line);
                return line;
            }

            if (process == null || !process.isAlive()) {
                throw new IllegalStateException("Bot process exited unexpectedly for player " + playerID +
                        exitCodeSuffix());
            }

            if (System.currentTimeMillis() > deadlineMillis) {
                throw new IllegalStateException("Timed out waiting for external bot action after " +
                        options.actionTimeoutMillis + "ms for player " + playerID + ".");
            }

            try {
                Thread.sleep(5L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw e;
            }
        }
    }

    private long computeDeadlineMillis(ElapsedCpuTimer ect) {
        long maxWait = options.actionTimeoutMillis;
        long remainingTurnBudget = Math.max(0L, ect.remainingTimeMillis());
        if (remainingTurnBudget > 0L) {
            maxWait = Math.min(maxWait, remainingTurnBudget);
        }
        return System.currentTimeMillis() + Math.max(1L, maxWait);
    }

    private void ensureMatchWithinLifetime() {
        if (options.matchTimeoutMillis <= 0L || matchStartMillis <= 0L) {
            return;
        }

        long elapsed = System.currentTimeMillis() - matchStartMillis;
        if (elapsed > options.matchTimeoutMillis) {
            throw new IllegalStateException("External bot exceeded match lifetime budget of " +
                    options.matchTimeoutMillis + "ms for player " + playerID + ".");
        }
    }

    private Action fallbackActionFromRanking(
            JSONObject request,
            JSONObject response,
            ArrayList<Action> legalActions,
            ArrayList<String> actionIds
    ) {
        JSONArray rankedActionIds = response.optJSONArray("rankedActionIds");
        if (rankedActionIds == null) {
            rankedActionIds = response.optJSONArray("ranked_action_ids");
        }
        if (rankedActionIds == null || rankedActionIds.isEmpty()) {
            return null;
        }

        for (int i = 0; i < rankedActionIds.length(); i++) {
            String candidateId = rankedActionIds.optString(i, null);
            if (candidateId == null || candidateId.isEmpty()) {
                continue;
            }
            int fallbackIdx = actionIds.indexOf(candidateId);
            if (fallbackIdx >= 0 && fallbackIdx < legalActions.size()) {
                Path debugPath = writeInvalidActionFallbackDebug(request, response, actionIds, candidateId);
                System.err.println("[ExternalProcessAgent] Falling back to ranked legal action for player " + playerID +
                        ": selected=" + response.optString("actionId", response.optString("action_id", null)) +
                        ", fallback=" + candidateId +
                        ", debug=" + debugPath);
                return legalActions.get(fallbackIdx);
            }
        }
        return null;
    }

    private Path writeInvalidActionFallbackDebug(
            JSONObject request,
            JSONObject response,
            ArrayList<String> actionIds,
            String fallbackActionId
    ) {
        try {
            File baseDir = options.workingDirectory != null ? options.workingDirectory : new File(System.getProperty("user.dir"));
            Path debugRoot = baseDir.toPath().resolve("debug-logs").resolve("invalid-action-fallbacks");
            Files.createDirectories(debugRoot);
            long nowMs = System.currentTimeMillis();
            Path out = debugRoot.resolve("external_process_invalid_action_" + nowMs + "_p" + playerID + "_r" + requestCounter + ".json");
            JSONObject payload = new JSONObject();
            payload.put("source", "ExternalProcessAgent");
            payload.put("created_at_ms", nowMs);
            payload.put("player_id", playerID);
            payload.put("command", command);
            payload.put("fallback_action_id", fallbackActionId);
            payload.put("legal_action_ids", actionIds);
            payload.put("request", request);
            payload.put("response", response);
            Files.writeString(out, payload.toString(2), StandardCharsets.UTF_8);
            return out;
        } catch (IOException e) {
            System.err.println("[ExternalProcessAgent] Failed to write invalid-action fallback debug log for player " +
                    playerID + ": " + e.getMessage());
            return null;
        }
    }

    private String exitCodeSuffix() {
        if (process == null) {
            return "";
        }
        try {
            return " (exit code " + process.exitValue() + ")";
        } catch (IllegalThreadStateException ignored) {
            return "";
        }
    }

    private void destroyProcess() {
        closeQuietly(processIn);
        closeQuietly(processOut);
        processIn = null;
        processOut = null;

        if (process != null) {
            process.destroy();
            if (process.isAlive()) {
                process.destroyForcibly();
            }
            process = null;
        }
        closeQuietly(stdoutLogWriter);
        closeQuietly(stderrLogWriter);
        stdoutLogWriter = null;
        stderrLogWriter = null;
    }

    private void awaitProcessExitAfterResult() {
        try {
            if (processIn != null) {
                processIn.close();
                processIn = null;
            }

            if (process != null && process.isAlive()) {
                boolean exited = process.waitFor(Math.max(1L, options.shutdownTimeoutMillis), java.util.concurrent.TimeUnit.MILLISECONDS);
                if (!exited) {
                    process.destroy();
                    if (!process.waitFor(Math.max(1L, options.shutdownTimeoutMillis / 2L), java.util.concurrent.TimeUnit.MILLISECONDS)) {
                        process.destroyForcibly();
                    }
                }
            }
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            System.err.println("[ExternalProcessAgent] Error while waiting for bot shutdown for player " + playerID +
                    ": " + e.getMessage());
        }
    }

    private void closeQuietly(Closeable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (IOException ignored) {
        }
    }

    private void openLogs() throws IOException {
        if (options.logRoot == null) {
            return;
        }

        Files.createDirectories(options.logRoot);
        stdoutLogWriter = Files.newBufferedWriter(options.logRoot.resolve(options.logPrefix + ".stdout.log"), StandardCharsets.UTF_8);
        stderrLogWriter = Files.newBufferedWriter(options.logRoot.resolve(options.logPrefix + ".stderr.log"), StandardCharsets.UTF_8);
    }

    private void logStdout(String line) {
        writeLog(stdoutLogWriter, line);
    }

    private void logStderr(String line) {
        writeLog(stderrLogWriter, line);
    }

    private void writeLog(BufferedWriter writer, String line) {
        if (writer == null) {
            return;
        }
        synchronized (writer) {
            try {
                writer.write(line);
                writer.newLine();
                writer.flush();
            } catch (IOException ignored) {
            }
        }
    }

    private static boolean disablesForwardModel(List<String> command) {
        for (String part : command) {
            if ("--no-forward-model".equals(part)) {
                return true;
            }
        }
        return false;
    }

    public static class Options {
        public long startupTimeoutMillis = 5000L;
        public long actionTimeoutMillis = 5000L;
        public long shutdownTimeoutMillis = 30000L;
        public long matchTimeoutMillis = 0L;
        public File workingDirectory = new File(System.getProperty("user.dir"));
        public Path logRoot = null;
        public String logPrefix = "external-bot";
        public Map<String, String> environment = new HashMap<>();

        public Options copy() {
            Options copy = new Options();
            copy.startupTimeoutMillis = startupTimeoutMillis;
            copy.actionTimeoutMillis = actionTimeoutMillis;
            copy.shutdownTimeoutMillis = shutdownTimeoutMillis;
            copy.matchTimeoutMillis = matchTimeoutMillis;
            copy.workingDirectory = workingDirectory == null ? null : new File(workingDirectory.getPath());
            copy.logRoot = logRoot;
            copy.logPrefix = logPrefix;
            copy.environment = new HashMap<>(environment);
            return copy;
        }

        public Options withLogRoot(Path root, String prefix) {
            Options copy = copy();
            copy.logRoot = root;
            copy.logPrefix = Objects.requireNonNullElse(prefix, copy.logPrefix);
            return copy;
        }
    }
}
