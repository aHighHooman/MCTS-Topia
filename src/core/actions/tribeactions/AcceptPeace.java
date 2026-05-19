package core.actions.tribeactions;

import core.Diplomacy;
import core.Types;
import core.actions.Action;
import core.game.GameState;

public class AcceptPeace extends TribeAction {

    private int targetID;

    public AcceptPeace(int tribeId) {
        super(Types.ACTION.ACCEPT_PEACE);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) { this.targetID = targetID; }
    public int getTargetID() { return this.targetID; }

    @Override
    public boolean isFeasible(final GameState gs) {
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        return tribeId != targetID &&
                diplomacy.hasPendingOffer(targetID, tribeId, Types.RELATIONSHIP.PEACE);
    }

    @Override
    public Action copy() {
        AcceptPeace acceptPeace = new AcceptPeace(this.tribeId);
        acceptPeace.setTargetID(targetID);
        return acceptPeace;
    }

    public String toString() {
        return "ACCEPT_PEACE by tribe " + this.tribeId + " from tribe " + this.targetID;
    }
}
