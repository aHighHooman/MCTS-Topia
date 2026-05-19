package core.actions.unitactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Infiltrate;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;

public class InfiltrateCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        Infiltrate action = (Infiltrate) a;
        if (!action.isFeasible(gs)) {
            return false;
        }

        Unit cloak = (Unit) gs.getActor(action.getUnitId());
        City targetCity = (City) gs.getActor(action.getTargetCityId());
        Board board = gs.getBoard();
        Tribe attacker = gs.getTribe(cloak.getTribeId());
        Tribe defender = gs.getTribe(targetCity.getTribeId());

        board.getDiplomacy().clearPendingOffersBetween(attacker.getTribeId(), defender.getTribeId());
        board.destroyEmbassiesBetween(gs, attacker.getTribeId(), defender.getTribeId());
        board.getDiplomacy().setRelationship(attacker.getTribeId(), defender.getTribeId(), Types.RELATIONSHIP.WAR, gs.getTick());

        Vector2d cityPos = targetCity.getPosition();
        Unit cityGuard = board.getUnitAt(cityPos.x, cityPos.y);
        if (cityGuard != null && cityGuard.getTribeId() == defender.getTribeId()) {
            cityGuard.setCurrentHP(cityGuard.getCurrentHPExact() - cloak.ATK);
            if (cityGuard.getCurrentHPExact() <= 0) {
                gs.killUnit(cityGuard);
            }
        }

        attacker.addStars(targetCity.getProduction());
        targetCity.setInfiltrated(true);
        spawnDaggers(gs, targetCity, attacker.getTribeId(), Math.min(5, targetCity.getLevel()));
        gs.killUnit(cloak);
        return true;
    }

    private void spawnDaggers(GameState gs, City targetCity, int tribeId, int numDaggers) {
        Board board = gs.getBoard();
        LinkedList<Vector2d> positions = board.getCityTiles(targetCity.getActorId());
        ArrayList<Vector2d> ordered = new ArrayList<>(positions);
        Vector2d cityPos = targetCity.getPosition();
        Tribe tribe = gs.getTribe(tribeId);
        Collections.sort(ordered, Comparator
                .comparingInt((Vector2d pos) -> spawnPriority(board, cityPos, pos, tribe))
                .thenComparingInt(pos -> pos.x)
                .thenComparingInt(pos -> pos.y));

        int spawned = 0;
        for (Vector2d pos : ordered) {
            if (spawned >= numDaggers) {
                break;
            }

            Types.UNIT spawnType = getSpawnType(board, pos, tribeId);
            if (spawnType == null) {
                continue;
            }

            Unit dagger = Types.UNIT.createUnit(pos.copy(), 0, false, -1, tribeId, spawnType);
            dagger.setStatus(Types.TURN_STATUS.FINISHED);
            board.addUnit(targetCity, dagger);
            gs.getTribe(tribeId).addScore(spawnType.getPoints());
            spawned++;
        }
    }

    private Types.UNIT getSpawnType(Board board, Vector2d pos, int tribeId) {
        if (board.getUnitAt(pos.x, pos.y) != null || !board.traversable(pos.x, pos.y, tribeId)) {
            return null;
        }

        Types.TERRAIN terrain = board.getTerrainAt(pos.x, pos.y);
        if (terrain == Types.TERRAIN.SHALLOW_WATER || terrain == Types.TERRAIN.DEEP_WATER) {
            return Types.UNIT.PIRATE;
        }

        return Types.UNIT.DAGGER;
    }

    private int spawnPriority(Board board, Vector2d cityPos, Vector2d pos, Tribe tribe) {
        if (cityPos.equals(pos)) {
            return 0;
        }

        Types.TERRAIN terrain = board.getTerrainAt(pos.x, pos.y);
        boolean getsDefenceBonus =
                (terrain == Types.TERRAIN.FOREST && tribe.getTechTree().isResearched(Types.TECHNOLOGY.ARCHERY)) ||
                (terrain == Types.TERRAIN.MOUNTAIN && tribe.getTechTree().isResearched(Types.TECHNOLOGY.CLIMBING)) ||
                (terrain.isWater() && tribe.getTechTree().isResearched(Types.TECHNOLOGY.AQUATISM));

        return getsDefenceBonus ? 1 : 2;
    }
}
