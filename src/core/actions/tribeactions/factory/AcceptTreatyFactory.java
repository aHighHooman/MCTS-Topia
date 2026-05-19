package core.actions.tribeactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.tribeactions.AcceptTreaty;
import core.actors.Actor;
import core.game.GameState;

import java.util.LinkedList;

public class AcceptTreatyFactory implements ActionFactory {
    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        return TargetedTribeActionFactorySupport.computeFeasibleActions(actor, gs, (tribeId, targetId) -> {
            AcceptTreaty action = new AcceptTreaty(tribeId);
            action.setTargetID(targetId);
            return action;
        });
    }
}
