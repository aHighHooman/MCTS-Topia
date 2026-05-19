package core.game;

import core.Constants;
import core.Types;
import core.actors.Tribe;
import players.Agent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeSet;

public final class HeadlessEvaluation {

    private HeadlessEvaluation() {
    }

    public static MatchResult runMatch(Agent[] players, Types.TRIBE[] tribes, long levelSeed, long gameSeed,
                                       Types.GAME_MODE gameMode) {
        return runMatch(new ArrayList<>(Arrays.asList(players)), tribes, levelSeed, gameSeed, gameMode);
    }

    public static MatchResult runMatch(ArrayList<Agent> players, Types.TRIBE[] tribes, long levelSeed, long gameSeed,
                                       Types.GAME_MODE gameMode) {
        return runConfiguredMatch(players, game -> game.init(players, levelSeed, tribes, gameSeed, gameMode));
    }

    public static MatchResult runMatch(Agent[] players, String[] levelLines, long gameSeed, Types.GAME_MODE gameMode) {
        return runMatch(new ArrayList<>(Arrays.asList(players)), levelLines, gameSeed, gameMode);
    }

    public static MatchResult runMatch(ArrayList<Agent> players, String[] levelLines, long gameSeed,
                                       Types.GAME_MODE gameMode) {
        try {
            Path levelFile = Files.createTempFile("tribes-headless-eval-", ".txt");
            try {
                Files.write(levelFile, Arrays.asList(levelLines), StandardCharsets.UTF_8);
                return runConfiguredMatch(players, game -> game.init(players, levelFile.toString(), gameSeed, gameMode));
            } finally {
                Files.deleteIfExists(levelFile);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create temporary level file for headless evaluation.", e);
        }
    }

    private static MatchResult runConfiguredMatch(ArrayList<Agent> players, GameInitializer initializer) {
        boolean visuals = Constants.VISUALS;
        boolean verbose = Constants.VERBOSE;
        boolean logStats = Constants.LOG_STATS;
        try {
            Constants.VISUALS = false;
            Constants.VERBOSE = false;
            Constants.LOG_STATS = false;

            Game game = new Game();
            initializer.init(game);
            game.run(null, null);

            MatchResult result = MatchResult.fromGame(game);
            result.validate(players.size());
            return result;
        } finally {
            Constants.VISUALS = visuals;
            Constants.VERBOSE = verbose;
            Constants.LOG_STATS = logStats;
        }
    }

    public static final class MatchResult {
        private final int tick;
        private final boolean gameOver;
        private final Types.TRIBE[] tribes;
        private final Types.RESULT[] winnerStatus;
        private final int[] scores;
        private final int[] rankingOrder;

        private MatchResult(int tick, boolean gameOver, Types.TRIBE[] tribes, Types.RESULT[] winnerStatus, int[] scores,
                            int[] rankingOrder) {
            this.tick = tick;
            this.gameOver = gameOver;
            this.tribes = tribes.clone();
            this.winnerStatus = winnerStatus.clone();
            this.scores = scores.clone();
            this.rankingOrder = rankingOrder.clone();
        }

        static MatchResult fromGame(Game game) {
            Tribe[] tribeObjects = game.getBoard().getTribes();
            Types.TRIBE[] tribes = new Types.TRIBE[tribeObjects.length];
            for (int i = 0; i < tribeObjects.length; i++) {
                tribes[i] = tribeObjects[i].getType();
            }
            return new MatchResult(
                    game.getTick(),
                    game.isGameOver(),
                    tribes,
                    game.getWinnerStatus(),
                    game.getScores(),
                    extractRankingOrder(game.getCurrentRanking())
            );
        }

        private static int[] extractRankingOrder(TreeSet<TribeResult> ranking) {
            int[] order = new int[ranking.size()];
            int index = 0;
            for (TribeResult tribeResult : ranking) {
                order[index++] = tribeResult.getId();
            }
            return order;
        }

        private void validate(int expectedPlayers) {
            if (!gameOver) {
                throw new IllegalStateException("Headless match ended without reaching a terminal state.");
            }
            if (tick <= 0) {
                throw new IllegalStateException("Headless match ended without advancing the game clock.");
            }
            if (winnerStatus.length != expectedPlayers) {
                throw new IllegalStateException("Winner array length mismatch.");
            }
            if (scores.length != expectedPlayers) {
                throw new IllegalStateException("Score array length mismatch.");
            }
            if (rankingOrder.length != expectedPlayers) {
                throw new IllegalStateException("Ranking length mismatch.");
            }

            boolean foundWinner = false;
            boolean[] seen = new boolean[expectedPlayers];
            for (Types.RESULT result : winnerStatus) {
                if (result == Types.RESULT.INCOMPLETE) {
                    throw new IllegalStateException("Terminal match still reports INCOMPLETE tribe results.");
                }
                if (result == Types.RESULT.WIN) {
                    foundWinner = true;
                }
            }

            for (int tribeId : rankingOrder) {
                if (tribeId < 0 || tribeId >= expectedPlayers) {
                    throw new IllegalStateException("Ranking references an invalid tribe id: " + tribeId);
                }
                if (seen[tribeId]) {
                    throw new IllegalStateException("Ranking contains duplicate tribe id: " + tribeId);
                }
                seen[tribeId] = true;
            }

            if (!foundWinner) {
                throw new IllegalStateException("Terminal match did not report any winner.");
            }
        }

        public int getTick() {
            return tick;
        }

        public boolean isGameOver() {
            return gameOver;
        }

        public Types.TRIBE[] getTribes() {
            return tribes.clone();
        }

        public Types.RESULT[] getWinnerStatus() {
            return winnerStatus.clone();
        }

        public int[] getScores() {
            return scores.clone();
        }

        public int[] getRankingOrder() {
            return rankingOrder.clone();
        }

        public String summary() {
            StringBuilder builder = new StringBuilder();
            builder.append("tick=").append(tick);
            builder.append(", ranking=");
            for (int i = 0; i < rankingOrder.length; i++) {
                if (i > 0) {
                    builder.append(" > ");
                }
                int tribeId = rankingOrder[i];
                builder.append(tribes[tribeId]).append("(").append(scores[tribeId]).append(", ").append(winnerStatus[tribeId]).append(")");
            }
            return builder.toString();
        }

        @Override
        public String toString() {
            return summary();
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof MatchResult)) {
                return false;
            }
            MatchResult other = (MatchResult) o;
            return tick == other.tick
                    && gameOver == other.gameOver
                    && Arrays.equals(tribes, other.tribes)
                    && Arrays.equals(winnerStatus, other.winnerStatus)
                    && Arrays.equals(scores, other.scores)
                    && Arrays.equals(rankingOrder, other.rankingOrder);
        }

        @Override
        public int hashCode() {
            int result = Integer.hashCode(tick);
            result = 31 * result + Boolean.hashCode(gameOver);
            result = 31 * result + Arrays.hashCode(tribes);
            result = 31 * result + Arrays.hashCode(winnerStatus);
            result = 31 * result + Arrays.hashCode(scores);
            result = 31 * result + Arrays.hashCode(rankingOrder);
            return result;
        }
    }

    @FunctionalInterface
    private interface GameInitializer {
        void init(Game game);
    }
}
