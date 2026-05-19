package core.actions.tribeactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.ProposePeace;
import core.game.GameState;

public class ProposePeaceCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        ProposePeace action = (ProposePeace) a;
        if (action.isFeasible(gs)) {
            gs.getBoard().getDiplomacy().proposeRelationship(action.getTribeId(), action.getTargetID(), Types.RELATIONSHIP.PEACE);
            return true;
        }
        return false;
    }
}
