package core.game;

import core.Constants;
import core.Types;
import players.Agent;
import players.ExternalProcessAgent;

import java.util.ArrayList;
import java.util.Arrays;

public final class DebugExternalMatch {

    private DebugExternalMatch() {
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Usage: DebugExternalMatch <levelSeed> <gameSeed> <agentSeed>");
            System.exit(1);
        }

        long levelSeed = Long.parseLong(args[0]);
        long gameSeed = Long.parseLong(args[1]);
        long agentSeed = Long.parseLong(args[2]);

        ArrayList<Agent> players = new ArrayList<>();
        players.add(new ExternalProcessAgent(agentSeed, new ArrayList<>(Arrays.asList("python", "py/bots/native_static_mcts_bot.py"))));
        players.add(new ExternalProcessAgent(agentSeed, new ArrayList<>(Arrays.asList("python", "py/bots/simple_bot.py"))));

        Constants.VISUALS = false;
        Constants.VERBOSE = true;

        Game game = new Game();
        game.init(players, levelSeed, new Types.TRIBE[]{Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS}, gameSeed, Types.GAME_MODE.CAPITALS);
        game.run(null, null);
    }
}
