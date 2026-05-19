package core.actions.unitactions.factory;

import core.Types;
import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.unitactions.Infiltrate;
import core.actors.Actor;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.LinkedList;

public class InfiltrateFactory implements ActionFactory {
    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        LinkedList<Action> infiltrations = new LinkedList<>();
        Unit unit = (Unit) actor;
        if ((unit.getType() != Types.UNIT.CLOAK && unit.getType() != Types.UNIT.DINGHY) || !unit.canAttack()) {
            return infiltrations;
        }

        Board board = gs.getBoard();
        for (Vector2d tile : unit.getPosition().neighborhood(unit.RANGE, 0, board.getSize())) {
            if (board.getTerrainAt(tile.x, tile.y) != Types.TERRAIN.CITY) {
                continue;
            }

            City city = board.getCityInBorders(tile.x, tile.y);
            if (city == null || city.getTribeId() == unit.getTribeId()) {
                continue;
            }

            Infiltrate action = new Infiltrate(unit.getActorId());
            action.setTargetCityId(city.getActorId());
            if (action.isFeasible(gs)) {
                infiltrations.add(action);
            }
        }

        return infiltrations;
    }
}
