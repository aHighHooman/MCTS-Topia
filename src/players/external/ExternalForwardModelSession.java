package players.external;

import core.actions.Action;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ExternalForwardModelSession {

    private final class SessionState {
        private final String stateId;
        private final GameState state;
        private final ArrayList<Action> actions;
        private final ArrayList<String> actionIds;

        private SessionState(String stateId, GameState state) {
            this(stateId, state, state.getAllAvailableActions(), null);
        }

        private SessionState(String stateId, GameState state, ArrayList<Action> actions, ArrayList<String> actionIds) {
            this.stateId = stateId;
            this.state = state;
            this.actions = actions;
            this.actionIds = actionIds != null ? actionIds : buildActionIds(stateId, actions.size());
        }

        private JSONObject toPayload(int playerId) {
            JSONObject payload = ExternalBotPayloadBuilder.buildForwardModelStatePayload(
                    state, playerId, actions, actionIds);
            payload.put("state_id", stateId);
            return payload;
        }

        private int resolveActionIndex(JSONObject request) {
            String actionId = request.optString("action_id", request.optString("actionId", null));
            if (actionId != null && !actionId.isEmpty()) {
                return actionIds.indexOf(actionId);
            }

            if (request.has("i")) {
                return request.optInt("i", -1);
            }
            if (request.has("action_index")) {
                return request.optInt("action_index", -1);
            }
            return request.optInt("actionIndex", -1);
        }

        private Action resolveAction(JSONObject request) {
            int actionIndex = resolveActionIndex(request);
            if (actionIndex < 0 || actionIndex >= actions.size()) {
                return null;
            }
            return actions.get(actionIndex);
        }
    }

    private final int playerId;
    private final Map<String, SessionState> states;
    private int nextStateIdx;

    public ExternalForwardModelSession(GameState rootState, int playerId,
                                       ArrayList<Action> rootActions, ArrayList<String> rootActionIds) {
        this.playerId = playerId;
        this.states = new HashMap<>();
        this.nextStateIdx = 1;
        registerState(ExternalBotPayloadBuilder.ROOT_STATE_ID, rootState,
                new ArrayList<>(rootActions), new ArrayList<>(rootActionIds));
    }

    public String getRootStateId() {
        return ExternalBotPayloadBuilder.ROOT_STATE_ID;
    }

    public JSONObject handleCommand(JSONObject request) {
        String command = request.optString("command", request.optString("fm_command", null));
        if (command == null || command.isEmpty()) {
            return withRequestMetadata(error(null, "Missing forward-model command."), request);
        }

        JSONObject response;
        switch (command) {
            case "inspect":
                response = inspect(request);
                break;
            case "step":
                response = step(request);
                break;
            case "step_many":
                response = stepMany(request);
                break;
            case "step_batch":
                response = stepBatch(request);
                break;
            case "release":
                response = release(request);
                break;
            default:
                response = error(command, "Unsupported forward-model command: " + command);
                break;
        }
        return withRequestMetadata(response, request);
    }

    private JSONObject inspect(JSONObject request) {
        String stateId = request.optString("state_id", getRootStateId());
        SessionState sessionState = states.get(stateId);
        if (sessionState == null) {
            return error("inspect", "Unknown state_id: " + stateId);
        }

        JSONObject response = ok("inspect");
        response.put("state", sessionState.toPayload(playerId));
        return response;
    }

    private JSONObject step(JSONObject request) {
        StepResult result = executeStep(request);
        if (result.errorResponse != null) {
            return result.errorResponse;
        }

        JSONObject response = ok("step");
        response.put("source_state_id", result.sourceStateId);
        response.put("action_index", result.actionIndex);
        response.put("action_id", result.actionId);
        response.put("state", result.childState.toPayload(playerId));
        return response;
    }

    private JSONObject stepMany(JSONObject request) {
        String sourceStateId = request.optString("state_id", getRootStateId());
        SessionState sourceState = states.get(sourceStateId);
        if (sourceState == null) {
            return error("step_many", "Unknown state_id: " + sourceStateId);
        }

        JSONArray rawActionIds = request.optJSONArray("action_ids");
        JSONArray rawActionIndexes = request.optJSONArray("action_indexes");
        if (rawActionIds == null && rawActionIndexes == null) {
            return error("step_many", "Expected action_ids or action_indexes for step_many.");
        }

        JSONArray results = new JSONArray();
        int count = rawActionIds != null ? rawActionIds.length() : rawActionIndexes.length();
        for (int i = 0; i < count; i++) {
            JSONObject childRequest = new JSONObject();
            childRequest.put("state_id", sourceStateId);
            if (rawActionIds != null) {
                childRequest.put("action_id", rawActionIds.optString(i, null));
            } else {
                childRequest.put("action_index", rawActionIndexes.optInt(i, -1));
            }

            StepResult result = executeStep(childRequest);
            JSONObject entry = new JSONObject();
            if (result.errorResponse != null) {
                entry.put("ok", false);
                entry.put("error", result.errorResponse.optString("message", "Unknown step_many error."));
                if (childRequest.has("action_id")) {
                    entry.put("action_id", childRequest.get("action_id"));
                }
                if (childRequest.has("action_index")) {
                    entry.put("action_index", childRequest.getInt("action_index"));
                }
            } else {
                entry.put("ok", true);
                entry.put("source_state_id", result.sourceStateId);
                entry.put("action_index", result.actionIndex);
                entry.put("action_id", result.actionId);
                entry.put("state", result.childState.toPayload(playerId));
            }
            results.put(entry);
        }

        JSONObject response = ok("step_many");
        response.put("source_state_id", sourceStateId);
        response.put("results", results);
        return response;
    }

    private JSONObject stepBatch(JSONObject request) {
        JSONArray rawRequests = request.optJSONArray("requests");
        if (rawRequests == null) {
            return error("step_batch", "Expected requests array for step_batch.");
        }

        JSONArray results = new JSONArray();
        for (int i = 0; i < rawRequests.length(); i++) {
            JSONObject childRequest = rawRequests.optJSONObject(i);
            if (childRequest == null) {
                JSONObject entry = new JSONObject();
                entry.put("ok", false);
                entry.put("error", "step_batch request entry must be an object.");
                results.put(entry);
                continue;
            }

            StepResult result = executeStep(childRequest);
            JSONObject entry = new JSONObject();
            if (result.errorResponse != null) {
                entry.put("ok", false);
                entry.put("error", result.errorResponse.optString("message", "Unknown step_batch error."));
                if (childRequest.has("state_id")) {
                    entry.put("source_state_id", childRequest.get("state_id"));
                }
                if (childRequest.has("action_id")) {
                    entry.put("action_id", childRequest.get("action_id"));
                }
                if (childRequest.has("action_index")) {
                    entry.put("action_index", childRequest.getInt("action_index"));
                }
            } else {
                entry.put("ok", true);
                entry.put("source_state_id", result.sourceStateId);
                entry.put("action_index", result.actionIndex);
                entry.put("action_id", result.actionId);
                entry.put("state", result.childState.toPayload(playerId));
            }
            results.put(entry);
        }

        JSONObject response = ok("step_batch");
        response.put("results", results);
        return response;
    }

    private JSONObject release(JSONObject request) {
        JSONArray stateIds = request.optJSONArray("state_ids");
        if (stateIds == null && request.has("state_id")) {
            stateIds = new JSONArray().put(request.get("state_id"));
        }

        JSONArray released = new JSONArray();
        if (stateIds != null) {
            for (int i = 0; i < stateIds.length(); i++) {
                String stateId = stateIds.optString(i, null);
                if (stateId == null || stateId.isEmpty() || getRootStateId().equals(stateId)) {
                    continue;
                }
                if (states.remove(stateId) != null) {
                    released.put(stateId);
                }
            }
        }

        JSONObject response = ok("release");
        response.put("released_state_ids", released);
        response.put("retained_state_count", states.size());
        return response;
    }

    private SessionState registerState(GameState state) {
        return registerState("s" + nextStateIdx++, state);
    }

    private SessionState registerState(String stateId, GameState state) {
        SessionState sessionState = new SessionState(stateId, state);
        states.put(stateId, sessionState);
        return sessionState;
    }

    private SessionState registerState(String stateId, GameState state,
                                       ArrayList<Action> actions, ArrayList<String> actionIds) {
        SessionState sessionState = new SessionState(stateId, state, actions, actionIds);
        states.put(stateId, sessionState);
        return sessionState;
    }

    private StepResult executeStep(JSONObject request) {
        String sourceStateId = request.optString("state_id", getRootStateId());
        SessionState sourceState = states.get(sourceStateId);
        if (sourceState == null) {
            return StepResult.error(error("step", "Unknown state_id: " + sourceStateId));
        }

        int actionIndex = sourceState.resolveActionIndex(request);
        Action action = sourceState.resolveAction(request);
        if (action == null) {
            return StepResult.error(error("step", "Unknown or out-of-range action for state_id " + sourceStateId));
        }

        GameState nextState = sourceState.state.copy();
        nextState.advance(action.copy(), true);
        SessionState childState = registerState(nextState);
        return StepResult.success(sourceStateId, actionIndex, sourceState.actionIds.get(actionIndex), childState);
    }

    private JSONObject ok(String command) {
        JSONObject response = new JSONObject();
        response.put("type", "forward_model_result");
        response.put("command", command);
        response.put("player_id", playerId);
        return response;
    }

    private JSONObject error(String command, String message) {
        JSONObject response = new JSONObject();
        response.put("type", "forward_model_error");
        if (command != null) {
            response.put("command", command);
        }
        response.put("player_id", playerId);
        response.put("message", message);
        return response;
    }

    private JSONObject withRequestMetadata(JSONObject response, JSONObject request) {
        if (request.has("request_id")) {
            response.put("request_id", request.get("request_id"));
        }
        return response;
    }

    private static ArrayList<String> buildActionIds(String stateId, int numActions) {
        ArrayList<String> actionIds = new ArrayList<>(numActions);
        String prefix = ExternalBotPayloadBuilder.ROOT_STATE_ID.equals(stateId) ? "A" : stateId + "_A";
        for (int i = 0; i < numActions; i++) {
            actionIds.add(prefix + i);
        }
        return actionIds;
    }

    private static final class StepResult {
        private final String sourceStateId;
        private final int actionIndex;
        private final String actionId;
        private final SessionState childState;
        private final JSONObject errorResponse;

        private StepResult(String sourceStateId, int actionIndex, String actionId,
                           SessionState childState, JSONObject errorResponse) {
            this.sourceStateId = sourceStateId;
            this.actionIndex = actionIndex;
            this.actionId = actionId;
            this.childState = childState;
            this.errorResponse = errorResponse;
        }

        private static StepResult success(String sourceStateId, int actionIndex, String actionId, SessionState childState) {
            return new StepResult(sourceStateId, actionIndex, actionId, childState, null);
        }

        private static StepResult error(JSONObject errorResponse) {
            return new StepResult(null, -1, null, null, errorResponse);
        }
    }
}
