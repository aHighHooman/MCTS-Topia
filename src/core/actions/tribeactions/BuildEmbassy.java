package core.actions.tribeactions;

import core.Types;
import core.actions.Action;
import core.game.GameState;

public class BuildEmbassy extends TribeAction {

    private int targetID;

    public BuildEmbassy(int tribeId) {
        super(Types.ACTION.BUILD_EMBASSY);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) {
        this.targetID = targetID;
    }

    public int getTargetID() {
        return targetID;
    }

    @Override
    public boolean isFeasible(final GameState gs) {
        return gs.getBoard().canBuildEmbassy(tribeId, targetID);
    }

    @Override
    public Action copy() {
        BuildEmbassy buildEmbassy = new BuildEmbassy(this.tribeId);
        buildEmbassy.setTargetID(targetID);
        return buildEmbassy;
    }

    @Override
    public String toString() {
        return "BUILD_EMBASSY by tribe " + tribeId + " in capital of tribe " + targetID;
    }
}
