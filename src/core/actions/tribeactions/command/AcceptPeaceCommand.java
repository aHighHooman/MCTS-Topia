package core.actions.tribeactions.command;

import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.AcceptPeace;
import core.game.GameState;

public class AcceptPeaceCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        AcceptPeace action = (AcceptPeace) a;
        if (action.isFeasible(gs)) {
            gs.getBoard().getDiplomacy().acceptPendingRelationship(action.getTargetID(), action.getTribeId(), gs.getTick());
            gs.getBoard().recomputeTradeNetworkForTribe(action.getTribeId());
            gs.getBoard().recomputeTradeNetworkForTribe(action.getTargetID());
            return true;
        }
        return false;
    }
}
