package core.actions.tribeactions;

import core.Diplomacy;
import core.Types;
import core.actions.Action;
import core.game.GameState;

public class CancelTreaty extends TribeAction {

    private int targetID;

    public CancelTreaty(int tribeId) {
        super(Types.ACTION.CANCEL_TREATY);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) { this.targetID = targetID; }
    public int getTargetID() { return this.targetID; }

    @Override
    public boolean isFeasible(final GameState gs) {
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        return tribeId != targetID &&
                diplomacy.isTreaty(tribeId, targetID);
    }

    @Override
    public Action copy() {
        CancelTreaty cancelTreaty = new CancelTreaty(this.tribeId);
        cancelTreaty.setTargetID(targetID);
        return cancelTreaty;
    }

    public String toString() {
        return "CANCEL_TREATY by tribe " + this.tribeId + " with tribe " + this.targetID;
    }
}
