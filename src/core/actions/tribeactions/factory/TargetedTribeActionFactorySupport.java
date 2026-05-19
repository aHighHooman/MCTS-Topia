package core.actions.tribeactions.factory;

import core.actions.Action;
import core.actions.tribeactions.TribeAction;
import core.actors.Actor;
import core.actors.Tribe;
import core.game.GameState;

import java.util.LinkedList;

final class TargetedTribeActionFactorySupport {

    interface Builder {
        TribeAction create(int tribeId, int targetId);
    }

    private TargetedTribeActionFactorySupport() {}

    static LinkedList<Action> computeFeasibleActions(final Actor actor, final GameState gs, Builder builder) {
        LinkedList<Action> actions = new LinkedList<>();
        Tribe tribe = (Tribe) actor;
        int tribeId = tribe.getTribeId();

        for (int i = 0; i < gs.getBoard().getTribes().length; i++) {
            if (i == tribeId) {
                continue;
            }

            TribeAction action = builder.create(tribeId, i);
            if (action.isFeasible(gs)) {
                actions.add(action);
            }
        }

        return actions;
    }
}
