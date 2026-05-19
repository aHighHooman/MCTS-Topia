package core.actions.unitactions.command;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Move;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import static core.Types.TERRAIN.*;

public class MoveCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Move action = (Move)a;
        if(action.isFeasible(gs)) {
            int unitId = action.getUnitId();
            Vector2d destination = action.getDestination();
            Unit unit = (Unit) gs.getActor(unitId);
            Board board = gs.getBoard();
            Tribe tribe = gs.getTribe(unit.getTribeId());
            Types.UNIT originalType = unit.getType();
            Types.TERRAIN destinationTerrain = board.getTerrainAt(destination.x, destination.y);
            if (board.hasHiddenEnemyCloak(unit.getTribeId(), destination.x, destination.y)) {
                Unit hiddenCloak = board.getUnitAt(destination.x, destination.y);
                hiddenCloak.setHidden(false);
                return true;
            }
            if (board.getUnitAt(destination.x, destination.y) != null) {
                return false;
            }

            board.moveUnit(unit, unit.getPosition().x, unit.getPosition().y, destination.x, destination.y, gs.getRandomGenerator());

            boolean stompOnArrival = originalType == Types.UNIT.JUGGERNAUT;
            if(unit.getType().isWaterUnit()){
                if(destinationTerrain != Types.TERRAIN.SHALLOW_WATER && destinationTerrain != Types.TERRAIN.DEEP_WATER){ // && destinationTerrain != Types.TERRAIN.CITY){
                    board.disembark(unit, tribe, destination.x, destination.y);
                    unit = board.getUnitAt(destination.x, destination.y);
                }
            }else {
                if(board.getBuildingAt(destination.x, destination.y) == Types.BUILDING.PORT &&
                        board.canUsePort(unit.getTribeId(), destination.x, destination.y)){
                    stompOnArrival = originalType == Types.UNIT.SUPERUNIT;
                    board.embark(unit, tribe, destination.x, destination.y);
                    unit = board.getUnitAt(destination.x, destination.y);
                }
            }

            if (stompOnArrival) {
                applyJuggernautStomp(gs, unit);
            }

            unit.transitionToStatus(Types.TURN_STATUS.MOVED);
            if (unit.getType() == Types.UNIT.CLOAK || unit.getType() == Types.UNIT.DINGHY) {
                unit.setHidden(destinationTerrain != Types.TERRAIN.CITY && destinationTerrain != Types.TERRAIN.VILLAGE);
            }

            return true;
        }
        return false;
    }

    private void applyJuggernautStomp(GameState gs, Unit unit) {
        if (unit == null) {
            return;
        }

        Vector2d position = unit.getPosition();
        for (Vector2d adjacent : position.neighborhood(1, 0, gs.getBoard().getSize())) {
            Unit target = gs.getBoard().getUnitAt(adjacent.x, adjacent.y);
            if (target == null || target.getTribeId() == unit.getTribeId()) {
                continue;
            }

            double damage = getJuggernautSplashDamage(unit, target, gs);
            if (damage <= 0) {
                continue;
            }

            if (target.getCurrentHPExact() <= damage) {
                unit.addKill();
                gs.getTribe(unit.getTribeId()).addKill();
                gs.killUnit(target);
            } else {
                target.setCurrentHP(target.getCurrentHPExact() - damage);
            }
        }
    }

    private double getJuggernautSplashDamage(Unit attacker, Unit target, GameState gs) {
        double attackForce = TribesConfig.JUGGERNAUT_ATTACK * (attacker.getCurrentHPExact() / attacker.getMaxHP());
        double defenceForce = target.getDefenceValue() * (target.getCurrentHPExact() / target.getMaxHP());
        Vector2d targetPos = target.getPosition();
        Types.TERRAIN targetTerrain = gs.getBoard().getTerrainAt(targetPos.x, targetPos.y);
        Tribe targetTribe = gs.getTribe(target.getTribeId());

        if (targetTerrain == CITY) {
            int cityID = gs.getBoard().getCityIdAt(targetPos.x, targetPos.y);
            if (targetTribe.controlsCity(cityID) && target.getType().canFortify()) {
                core.actors.City city = (core.actors.City) gs.getActor(cityID);
                defenceForce *= city.hasWalls() ? TribesConfig.DEFENCE_IN_WALLS : TribesConfig.DEFENCE_BONUS;
            }
        } else if ((targetTerrain == MOUNTAIN && targetTribe.getTechTree().isResearched(Types.TECHNOLOGY.CLIMBING)) ||
                (targetTerrain == FOREST && targetTribe.getTechTree().isResearched(Types.TECHNOLOGY.ARCHERY)) ||
                (targetTerrain.isWater() && targetTribe.getTechTree().isResearched(Types.TECHNOLOGY.AQUATISM))) {
            defenceForce *= TribesConfig.DEFENCE_BONUS;
        }

        double totalDamage = attackForce + defenceForce;
        int attackResult = (int) Math.round((attackForce / totalDamage) * TribesConfig.JUGGERNAUT_ATTACK * TribesConfig.ATTACK_MODIFIER);
        return attackResult / 2.0;
    }
}
