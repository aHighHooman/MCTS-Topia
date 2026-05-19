import core.Constants;
import core.Types;
import core.game.Game;
import core.game.TribeResult;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.ExternalProcessAgent;
import utils.file.IO;
import utils.stats.MultiStatSummary;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

/**
 * Entry point of the framework.
 */
public class Tournament {

    private static final int DEFAULT_MATCH_RETRY_LIMIT = 2;
    private static final int DEFAULT_TURN_LIMIT = 256;
    private static final double DEFAULT_ELO = 1500.0;
    private static final double ELO_K_FACTOR = 24.0;
    private static final long[] DEFAULT_SEEDS = new long[]{1L};

    public static void main(String[] args) {
        Types.GAME_MODE gameMode = Types.GAME_MODE.MIGHT;
        Tournament t = new Tournament(gameMode);
        Constants.VERBOSE = true;
        Constants.LOG_STATS = false;

        JSONObject config = new IO().readJSON("tournament.json");
        if (args.length > 0) {
            config = new IO().readJSON(args[0]);
        }

        if (config == null || config.isEmpty()) {
            throw new IllegalStateException("Could not read tournament config. External participants must be configured explicitly.");
        } else {
            try {
                gameMode = Run.parseGameModeStr(config.getString("Game Mode"));
                t = new Tournament(gameMode);
                t.matchRetryLimit = Math.max(0, config.optInt("Match Retry Limit", DEFAULT_MATCH_RETRY_LIMIT));
                t.turnLimit = Math.max(0, config.optInt("Turn Limit", DEFAULT_TURN_LIMIT));
                if (config.has("Max Actions Per Turn")) {
                    Game.MAX_ACTIONS_PER_TURN = Math.max(0, config.getInt("Max Actions Per Turn"));
                }
                if (config.has("Max Actions Per Game")) {
                    Game.MAX_ACTIONS_PER_GAME = Math.max(0, config.getInt("Max Actions Per Game"));
                }
                if (config.has("External Log Dir")) {
                    t.externalLogRoot = Paths.get(config.getString("External Log Dir"));
                }

                JSONArray participantsArray = config.getJSONArray("Participants");
                int nPlayers = participantsArray.length();
                t.mapType = config.has("Map Type")
                        ? Run.parseMapTypeStr(config.getString("Map Type"))
                        : Run.DEFAULT_MAP_TYPE;
                t.mapSize = config.has("Map Size")
                        ? Run.parseMapSizeStr(config.getString("Map Size"))
                        : Run.defaultMapSizeForPlayers(nPlayers);
                t.gloryTargetScore = Run.parseGloryTargetScore(config);

                t.setParticipants(parseParticipants(participantsArray));
                Constants.VERBOSE = config.optBoolean("Verbose", false);
                Constants.LOG_STATS = config.optBoolean("Log Stats", false);
                t.RUN_VERBOSE = Constants.VERBOSE;
                t.writeMatchLogs = config.optBoolean("Write Match Logs", t.hasExternalParticipants());
                if (config.has("Level Seeds")) {
                    t.setSeeds(config.getJSONArray("Level Seeds"));
                }
            } catch (Exception e) {
                throw new IllegalStateException("Malformed tournament config file.", e);
            }
        }

        t.run();
    }

    private Types.GAME_MODE gameMode;
    private Types.MAP_SIZE mapSize = Types.MAP_SIZE.TINY;
    private Types.MAP_TYPE mapType = Run.DEFAULT_MAP_TYPE;
    private int gloryTargetScore = Run.DEFAULT_GLORY_TARGET_SCORE;
    private boolean RUN_VERBOSE = true;
    private boolean writeMatchLogs = false;
    private int matchRetryLimit = DEFAULT_MATCH_RETRY_LIMIT;
    private int turnLimit = DEFAULT_TURN_LIMIT;
    private Path externalLogRoot = Paths.get("logs", "tournament");
    private final HashMap<Integer, Participant> participants;
    private MultiStatSummary[] stats;
    private Types.TRIBE[] tribes;
    private long[] seeds;
    private double[] eloRatings;
    private double[] pStrongestProbabilities;
    private double[][] pairwisePoints;
    private final ArrayList<FailedMatch> failedMatches = new ArrayList<>();

    private Tournament(Types.GAME_MODE gameMode) {
        this.gameMode = gameMode;
        this.participants = new HashMap<>();
        this.seeds = DEFAULT_SEEDS.clone();
    }

    public void setParticipants(Participant[] participantEntries) {
        if (participantEntries == null || participantEntries.length == 0) {
            throw new IllegalArgumentException("Tournament requires at least one participant.");
        }
        participants.clear();
        stats = new MultiStatSummary[participantEntries.length];
        eloRatings = new double[participantEntries.length];
        pStrongestProbabilities = new double[participantEntries.length];
        pairwisePoints = new double[participantEntries.length][participantEntries.length];
        tribes = new Types.TRIBE[participantEntries.length];
        for (int i = 0; i < participantEntries.length; ++i) {
            Participant p = participantEntries[i];
            if (p == null) {
                throw new IllegalArgumentException("Participant " + i + " is null.");
            }
            participants.put(i, p);
            tribes[i] = p.initialTribe;
            stats[i] = initMultiStat(p);
            eloRatings[i] = DEFAULT_ELO;
        }
    }

    private void setSeeds(JSONArray seeds) {
        if (seeds == null || seeds.length() == 0) {
            this.seeds = DEFAULT_SEEDS.clone();
            return;
        }
        this.seeds = new long[seeds.length()];
        for (int i = 0; i < this.seeds.length; ++i) {
            this.seeds[i] = Long.parseLong(String.valueOf(seeds.get(i)));
        }
    }

    private boolean hasExternalParticipants() {
        for (Participant participant : participants.values()) {
            if (participant.playerType == Run.PlayerType.EXTERNAL) {
                return true;
            }
        }
        return false;
    }

    private String shortExternalLabel(int participantId) {
        Participant participant = participants.get(participantId);
        if (participant == null || participant.externalCommand == null || participant.externalCommand.isEmpty()) {
            return "external";
        }

        ArrayList<String> cmd = participant.externalCommand;

        String last = cmd.get(cmd.size() - 1).replace("\\", "/");
        int slash = last.lastIndexOf('/');
        if (slash >= 0) {
            last = last.substring(slash + 1);
        }

        return last;
    }

    private String participantDisplay(Participant p) {
        if (p == null) {
            return "unknown";
        }
        if (p.displayName != null && !p.displayName.isEmpty()) {
            return p.participantId + ":" + p.displayName + "(" + p.initialTribe + ")";
        }
        if (p.playerType == Run.PlayerType.EXTERNAL) {
            return p.participantId + ":" + p.playerType + ":" + shortExternalLabel(p.participantId)
                    + "(" + p.initialTribe + ")";
        }
        return p.participantId + ":" + p.playerType + "(" + p.initialTribe + ")";
    }

    private void run() {
        for (int seedIndex = 0; seedIndex < seeds.length; seedIndex++) {
            long configuredSeed = seeds[seedIndex];
            long matchSeed = configuredSeed;
            if (matchSeed == -1) {
                matchSeed = System.currentTimeMillis() + new Random().nextInt();
            }
            System.out.println("**** Playing level with seed " + matchSeed + " ****");

            MatchAssignment matchAssignment = buildAssignment();
            System.out.println(matchAssignment.displayLine(seedIndex, seeds.length));

            boolean completed = false;
            Throwable lastFailure = null;
            int maxAttempts = matchRetryLimit + 1;

            for (int attempt = 1; attempt <= maxAttempts && !completed; attempt++) {
                Path matchLogDir = matchLogDir(matchSeed, seedIndex, attempt);

                try {
                    if (writeMatchLogs) {
                        writeMatchMetadata(matchLogDir, matchSeed, matchAssignment);
                    }
                    Game game = prepareGame(matchSeed, matchAssignment, matchLogDir);
                    Run.runGame(game);
                    addGameResults(game, matchAssignment.participantsBySeat);
                    updateElo(game, matchAssignment.participantsBySeat);
                    completed = true;
                } catch (Exception e) {
                    lastFailure = e;
                    if (writeMatchLogs) {
                        writeFailureMetadata(matchLogDir, e);
                    }

                    if (Boolean.getBoolean("tribes.debug.stop_on_game_error")) {
                        throw new RuntimeException("Tournament aborted while running level seed " + matchSeed
                                + ", matchup " + Arrays.toString(matchAssignment.playerTypes)
                                + ", attempt " + attempt, e);
                    }

                    System.err.println("Error running game attempt " + attempt + "/" + maxAttempts
                            + ": " + e.getMessage());
                    if (attempt < maxAttempts) {
                        System.out.println("Error running a game, retrying attempt "
                                + (attempt + 1) + "/" + maxAttempts + ".");
                    }
                }
            }

            if (!completed) {
                FailedMatch failedMatch = new FailedMatch(matchSeed, matchAssignment, maxAttempts, lastFailure);
                failedMatches.add(failedMatch);
                System.out.println("Match failed after " + maxAttempts + " attempts: " + failedMatch.summary());
            }
        }

        printRunResults();
    }

    private MatchAssignment buildAssignment() {
        Participant[] participantsBySeat = new Participant[participants.size()];
        Run.PlayerType[] playerTypes = new Run.PlayerType[participants.size()];
        StringBuilder display = new StringBuilder();
        display.append("Playing with [");
        for (int seat = 0; seat < participants.size(); seat++) {
            Participant p = participants.get(seat);
            if (seat > 0) {
                display.append(", ");
            }
            display.append(participantDisplay(p));
            playerTypes[seat] = p.playerType;
            participantsBySeat[seat] = p;
        }
        display.append("]");
        return new MatchAssignment(playerTypes, participantsBySeat, display.toString());
    }

    private MultiStatSummary initMultiStat(Participant p) {
        MultiStatSummary mss = new MultiStatSummary(p);
        mss.registerVariable("v");
        mss.registerVariable("s");
        return mss;
    }

    private Game prepareGame(long matchSeed, MatchAssignment matchAssignment, Path matchLogDir) {
        if (RUN_VERBOSE) {
            System.out.println("Match seed: " + matchSeed);
        }

        ArrayList<Agent> players = getPlayers(matchAssignment, matchSeed, matchLogDir);
        Game game = new Game();

        if (RUN_VERBOSE) {
            System.out.println("Level seed: " + matchSeed);
        }

        game.init(players, matchSeed, tribes, matchSeed, gameMode, mapSize, mapType, gloryTargetScore);
        game.setTurnLimitOverride(turnLimit);
        return game;
    }

    private ArrayList<Agent> getPlayers(MatchAssignment matchAssignment, long matchSeed, Path matchLogDir) {
        ArrayList<Agent> players = new ArrayList<>();
        ArrayList<Integer> allIds = new ArrayList<>();
        for (int i = 0; i < matchAssignment.playerTypes.length; ++i) {
            allIds.add(i);
        }

        for (int i = 0; i < matchAssignment.playerTypes.length; ++i) {
            Participant participant = matchAssignment.participantsBySeat[i];
            ArrayList<String> externalCommand = participant == null ? null : participant.externalCommand;
            Run.AgentFactoryConfig perAgentConfig = participant == null
                    ? Run.AgentFactoryConfig.fromStatics()
                    : participant.agentFactoryConfig;
            if (matchAssignment.playerTypes[i] == Run.PlayerType.EXTERNAL) {
                Path baseLogRoot = perAgentConfig.externalOptions.logRoot == null
                        ? matchLogDir
                        : perAgentConfig.externalOptions.logRoot.resolve("seed-" + matchSeed);
                Path participantLogRoot = baseLogRoot.resolve(sanitizeLabel(participantDisplay(participant)));
                ExternalProcessAgent.Options options = perAgentConfig.externalOptions.withLogRoot(
                        participantLogRoot,
                        "player-" + i + "-" + sanitizeLabel(participantDisplay(participant))
                );
                perAgentConfig = perAgentConfig.withExternalOptions(options);
            }

            Agent ag = Run.getAgent(matchAssignment.playerTypes[i], matchSeed, externalCommand, perAgentConfig);
            ag.setPlayerIDs(i, allIds);
            players.add(ag);
        }

        return players;
    }

    private void addGameResults(Game game, Participant[] participantsBySeat) {
        TreeSet<TribeResult> ranking = game.getCurrentRanking();
        for (TribeResult tr : ranking) {
            int pId = participantsBySeat[tr.getId()].participantId;

            int victoryCount = tr.getResult() == Types.RESULT.WIN ? 1 : 0;
            stats[pId].getVariable("v").add(victoryCount);
            stats[pId].getVariable("s").add(tr.getScore());
        }
    }

    private void updateElo(Game game, Participant[] participantsBySeat) {
        TreeSet<TribeResult> ranking = game.getCurrentRanking();
        ArrayList<TribeResult> orderedResults = new ArrayList<>(ranking);
        int n = orderedResults.size();
        if (n < 2) {
            return;
        }

        double[] deltas = new double[eloRatings.length];
        double scale = ELO_K_FACTOR / (double) (n - 1);
        for (int i = 0; i < n; i++) {
            TribeResult a = orderedResults.get(i);
            int participantA = participantsBySeat[a.getId()].participantId;
            for (int j = i + 1; j < n; j++) {
                TribeResult b = orderedResults.get(j);
                int participantB = participantsBySeat[b.getId()].participantId;

                double actualA = compareResults(a, b);
                recordPairwiseOutcome(participantA, participantB, actualA);
                double expectedA = 1.0 / (1.0 + Math.pow(10.0, (eloRatings[participantB] - eloRatings[participantA]) / 400.0));
                double delta = scale * (actualA - expectedA);
                deltas[participantA] += delta;
                deltas[participantB] -= delta;
            }
        }

        for (int i = 0; i < eloRatings.length; i++) {
            eloRatings[i] += deltas[i];
        }
    }

    private double compareResults(TribeResult a, TribeResult b) {
        if (a.getResult() != b.getResult()) {
            return a.getResult() == Types.RESULT.WIN ? 1.0 : 0.0;
        }
        if (a.getScore() > b.getScore()) {
            return 1.0;
        }
        if (a.getScore() < b.getScore()) {
            return 0.0;
        }
        return 0.5;
    }

    private void recordPairwiseOutcome(int participantA, int participantB, double actualA) {
        pairwisePoints[participantA][participantB] += actualA;
        pairwisePoints[participantB][participantA] += (1.0 - actualA);
    }

    private void updatePStrongestProbabilities() {
        if (pStrongestProbabilities == null || pStrongestProbabilities.length == 0) {
            return;
        }

        Arrays.fill(pStrongestProbabilities, 0.0);
        if (pStrongestProbabilities.length == 1) {
            pStrongestProbabilities[0] = 1.0;
            return;
        }

        double[] logStrength = new double[pStrongestProbabilities.length];
        double maxLogStrength = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < pStrongestProbabilities.length; i++) {
            double logScore = 0.0;
            for (int j = 0; j < pStrongestProbabilities.length; j++) {
                if (i == j) {
                    continue;
                }
                double posteriorMeanWinProb = (1.0 + pairwisePoints[i][j])
                        / (2.0 + pairwisePoints[i][j] + pairwisePoints[j][i]);
                logScore += Math.log(posteriorMeanWinProb);
            }
            logStrength[i] = logScore;
            if (logScore > maxLogStrength) {
                maxLogStrength = logScore;
            }
        }

        double total = 0.0;
        for (int i = 0; i < pStrongestProbabilities.length; i++) {
            pStrongestProbabilities[i] = Math.exp(logStrength[i] - maxLogStrength);
            total += pStrongestProbabilities[i];
        }

        if (total == 0.0) {
            double uniform = 1.0 / pStrongestProbabilities.length;
            Arrays.fill(pStrongestProbabilities, uniform);
            return;
        }

        for (int i = 0; i < pStrongestProbabilities.length; i++) {
            pStrongestProbabilities[i] /= total;
        }
    }

    private void printRunResults() {
        if (stats == null) {
            return;
        }

        updatePStrongestProbabilities();

        Arrays.sort(stats, (o1, o2) -> {
            int p1 = ((Participant) o1.getOwner()).participantId;
            int p2 = ((Participant) o2.getOwner()).participantId;

            if (o1.getVariable("v").sum() > o2.getVariable("v").sum()) return -1;
            if (o1.getVariable("v").sum() < o2.getVariable("v").sum()) return 1;
            if (pStrongestProbabilities[p1] > pStrongestProbabilities[p2]) return -1;
            if (pStrongestProbabilities[p1] < pStrongestProbabilities[p2]) return 1;
            if (eloRatings[p1] > eloRatings[p2]) return -1;
            if (eloRatings[p1] < eloRatings[p2]) return 1;
            if (o1.getVariable("s").mean() > o2.getVariable("s").mean()) return -1;
            if (o1.getVariable("s").mean() < o2.getVariable("s").mean()) return 1;
            return Integer.compare(p1, p2);
        });

        System.out.println("--------- RESULTS ---------");
        for (MultiStatSummary stat : stats) {
            Participant thisParticipant = (Participant) stat.getOwner();
            int participantId = thisParticipant.participantId;
            int w = (int) stat.getVariable("v").sum();
            int n = stat.getVariable("v").n();
            double percW = n == 0 ? 0.0 : 100.0 * (double) w / n;

            System.out.printf("[N:%d];", n);
            System.out.printf("[%%:%.2f];", percW);
            System.out.printf("[W:%d];", w);
            System.out.printf("[P(strongest):%.2f%%];", pStrongestProbabilities[participantId] * 100.0);
            System.out.printf("[ELO:%.1f];", eloRatings[participantId]);
            System.out.printf("[Score:%.2f];", stat.getVariable("s").mean());
            System.out.printf("[Player:%s]", participantDisplay(thisParticipant));
            System.out.println();
        }

        if (!failedMatches.isEmpty()) {
            System.out.println("--------- FAILED MATCHES ---------");
            for (FailedMatch failedMatch : failedMatches) {
                System.out.println(failedMatch.summary());
            }
        }
    }

    private Path matchLogDir(long matchSeed, int seedIndex, int attempt) {
        return externalLogRoot
                .resolve("seed-" + matchSeed)
                .resolve("game-" + (seedIndex + 1))
                .resolve("attempt-" + attempt);
    }

    private void writeMatchMetadata(Path matchLogDir, long matchSeed, MatchAssignment matchAssignment) {
        try {
            Files.createDirectories(matchLogDir);
            JSONObject metadata = new JSONObject();
            metadata.put("matchSeed", matchSeed);
            metadata.put("levelSeed", matchSeed);
            metadata.put("gameSeed", matchSeed);
            metadata.put("agentSeed", matchSeed);
            metadata.put("gameMode", gameMode.toString());
            metadata.put("mapType", mapType.toString());
            metadata.put("mapSize", mapSize.toString());
            metadata.put("gloryTargetScore", gloryTargetScore);
            metadata.put("retryLimit", matchRetryLimit);

            JSONArray playersJson = new JSONArray();
            for (int i = 0; i < matchAssignment.playerTypes.length; i++) {
                JSONObject playerJson = new JSONObject();
                playerJson.put("seat", i);
                playerJson.put("tribe", tribes[i].toString());
                playerJson.put("playerType", matchAssignment.playerTypes[i].toString());

                Participant participant = matchAssignment.participantsBySeat[i];
                if (participant != null) {
                    playerJson.put("participantId", participant.participantId);
                    playerJson.put("participantDisplay", participantDisplay(participant));
                    if (participant.displayName != null && !participant.displayName.isEmpty()) {
                        playerJson.put("name", participant.displayName);
                    }
                    if (participant.externalCommand != null) {
                        playerJson.put("externalCommand", new JSONArray(participant.externalCommand));
                    }
                }
                playersJson.put(playerJson);
            }
            metadata.put("players", playersJson);

            Files.write(matchLogDir.resolve("match.json"),
                    metadata.toString(2).getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException("Failed to prepare tournament log directory " + matchLogDir, e);
        }
    }

    private void writeFailureMetadata(Path matchLogDir, Throwable error) {
        if (matchLogDir == null) {
            return;
        }
        try {
            Files.createDirectories(matchLogDir);
            JSONObject failure = new JSONObject();
            failure.put("errorType", error == null ? "unknown" : error.getClass().getName());
            failure.put("message", error == null ? "unknown failure" : String.valueOf(error.getMessage()));
            Files.write(matchLogDir.resolve("failure.json"),
                    failure.toString(2).getBytes(StandardCharsets.UTF_8));
        } catch (IOException ignored) {
        }
    }

    private String sanitizeLabel(String label) {
        if (label == null || label.isEmpty()) {
            return "player";
        }
        return label.replaceAll("[^A-Za-z0-9._-]+", "_");
    }

    private static void printRunHelp(String[] args) {
        System.out.print("Invalid Arguments ");
        for (String s : args) {
            System.out.print(s + " ");
        }
        System.out.println(". Usage: ");
        System.out.println("'java Tournament <jsonConfigFile>', where: ");
        System.out.println("\t<jsonConfigFile> is the JSON file with the tournament configuration.");
        System.out.println("Example: java -jar tournament.json");
    }

    private static Participant[] parseParticipants(JSONArray participantsArray) throws Exception {
        Participant[] parsed = new Participant[participantsArray.length()];
        for (int i = 0; i < participantsArray.length(); i++) {
            JSONObject participantJson = participantsArray.getJSONObject(i);
            Run.PlayerType playerType = Run.parsePlayerTypeStr(participantJson.getString("Type"));
            Types.TRIBE tribe = Run.parseTribeStr(participantJson.getString("Tribe"));
            ArrayList<String> externalCommand = parseParticipantCommand(participantJson);
            Run.AgentFactoryConfig config = Run.AgentFactoryConfig.fromConfig(participantJson);
            String displayName = participantJson.optString("Name", "");
            parsed[i] = new Participant(playerType, i, tribe, displayName, config, externalCommand);
        }
        return parsed;
    }

    private static ArrayList<String> parseParticipantCommand(JSONObject participantJson) throws Exception {
        if (!participantJson.has("External Command") || participantJson.isNull("External Command")) {
            return null;
        }

        return Run.parseExternalCommandValue(participantJson.get("External Command"));
    }

    private static final class MatchAssignment {
        private final Run.PlayerType[] playerTypes;
        private final Participant[] participantsBySeat;
        private final String displayPrefix;

        private MatchAssignment(Run.PlayerType[] playerTypes, Participant[] participantsBySeat,
                                String displayPrefix) {
            this.playerTypes = playerTypes;
            this.participantsBySeat = participantsBySeat;
            this.displayPrefix = displayPrefix;
        }

        private String displayLine(int seedIndex, int totalSeeds) {
            return displayPrefix + " (" + (seedIndex + 1) + "/" + totalSeeds + ")";
        }
    }

    private static final class Participant {
        private final Run.PlayerType playerType;
        private final int participantId;
        private final Types.TRIBE initialTribe;
        private final String displayName;
        private final Run.AgentFactoryConfig agentFactoryConfig;
        private final ArrayList<String> externalCommand;

        private Participant(Run.PlayerType playerType, int participantId, Types.TRIBE initialTribe,
                            String displayName, Run.AgentFactoryConfig agentFactoryConfig,
                            ArrayList<String> externalCommand) {
            this.playerType = playerType;
            this.participantId = participantId;
            this.initialTribe = initialTribe;
            this.displayName = displayName == null ? "" : displayName.trim();
            this.agentFactoryConfig = agentFactoryConfig;
            this.externalCommand = externalCommand == null ? null : new ArrayList<>(externalCommand);
        }
    }

    private final class FailedMatch {
        private final long levelSeed;
        private final Run.PlayerType[] playerTypes;
        private final Types.TRIBE[] matchupTribes;
        private final ArrayList<String> participantLabels;
        private final int attempts;
        private final String errorSummary;

        private FailedMatch(long levelSeed, MatchAssignment matchAssignment, int attempts, Throwable failure) {
            this.levelSeed = levelSeed;
            this.playerTypes = matchAssignment.playerTypes.clone();
            this.matchupTribes = tribes.clone();
            this.participantLabels = new ArrayList<>();
            for (Participant participant : matchAssignment.participantsBySeat) {
                participantLabels.add(participantDisplay(participant));
            }
            this.attempts = attempts;
            this.errorSummary = failure == null ? "unknown failure"
                    : failure.getClass().getSimpleName() + ": " + String.valueOf(failure.getMessage());
        }

        private String summary() {
            return "seed=" + levelSeed
                    + ", attempts=" + attempts
                    + ", players=" + Arrays.toString(playerTypes)
                    + ", tribes=" + Arrays.toString(matchupTribes)
                    + ", assignments=" + participantLabels
                    + ", error=" + errorSummary;
        }
    }
}
