package core.actions.tribeactions;

import core.Diplomacy;
import core.Types;
import core.actions.Action;
import core.game.GameState;

public class AcceptTreaty extends TribeAction {

    private int targetID;

    public AcceptTreaty(int tribeId) {
        super(Types.ACTION.ACCEPT_TREATY);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) { this.targetID = targetID; }
    public int getTargetID() { return this.targetID; }

    @Override
    public boolean isFeasible(final GameState gs) {
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        return tribeId != targetID &&
                diplomacy.hasPendingOffer(targetID, tribeId, Types.RELATIONSHIP.TREATY);
    }

    @Override
    public Action copy() {
        AcceptTreaty acceptTreaty = new AcceptTreaty(this.tribeId);
        acceptTreaty.setTargetID(targetID);
        return acceptTreaty;
    }

    public String toString() {
        return "ACCEPT_TREATY by tribe " + this.tribeId + " from tribe " + this.targetID;
    }
}
