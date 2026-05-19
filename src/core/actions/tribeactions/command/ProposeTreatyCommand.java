package core.actions.tribeactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.ProposeTreaty;
import core.game.GameState;

public class ProposeTreatyCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        ProposeTreaty action = (ProposeTreaty) a;
        if (action.isFeasible(gs)) {
            gs.getBoard().getDiplomacy().proposeRelationship(action.getTribeId(), action.getTargetID(), Types.RELATIONSHIP.TREATY);
            return true;
        }
        return false;
    }
}
