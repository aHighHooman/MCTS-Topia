package core.actions.tribeactions.command;

import core.TribesConfig;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.BuildEmbassy;
import core.actors.City;
import core.actors.Tribe;
import core.game.GameState;
import utils.Vector2d;

public class BuildEmbassyCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        BuildEmbassy action = (BuildEmbassy) a;
        if (!action.isFeasible(gs)) {
            return false;
        }

        int ownerTribeId = action.getTribeId();
        int hostTribeId = action.getTargetID();
        Tribe owner = gs.getTribe(ownerTribeId);
        City capital = gs.getBoard().getCapitalCity(hostTribeId);

        if (!gs.getBoard().buildEmbassy(gs, ownerTribeId, hostTribeId)) {
            return false;
        }

        owner.subtractStars(TribesConfig.EMBASSY_COST);
        Vector2d capitalPos = capital.getPosition();
        owner.revealArea(gs.getBoard(), capitalPos.x, capitalPos.y, 1);
        return true;
    }
}
