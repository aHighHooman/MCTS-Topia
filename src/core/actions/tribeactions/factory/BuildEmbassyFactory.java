package core.actions.tribeactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.tribeactions.BuildEmbassy;
import core.actors.Actor;
import core.game.GameState;

import java.util.LinkedList;

public class BuildEmbassyFactory implements ActionFactory {
    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        return TargetedTribeActionFactorySupport.computeFeasibleActions(actor, gs, (tribeId, targetId) -> {
            BuildEmbassy action = new BuildEmbassy(tribeId);
            action.setTargetID(targetId);
            return action;
        });
    }
}
