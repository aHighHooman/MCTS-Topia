package core.actions.unitactions.command;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Attack;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.GameState;
import utils.Pair;
import utils.Vector2d;

import static core.Types.TECHNOLOGY.*;
import static core.Types.TERRAIN.*;

public class AttackCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Attack action = (Attack) a;

        if (!action.isFeasible(gs)) {
            return false;
        }

        Unit attacker = (Unit) gs.getActor(action.getUnitId());
        Unit target = (Unit) gs.getActor(action.getTargetId());
        if (attacker == null || target == null) {
            return false;
        }

        attacker.transitionToStatus(Types.TURN_STATUS.ATTACKED);
        Tribe attackerTribe = gs.getTribe(attacker.getTribeId());
        attackerTribe.resetPacifistCount();
        gs.getBoard().getDiplomacy().clearPendingOffersBetween(attacker.getTribeId(), target.getTribeId());
        gs.getBoard().getDiplomacy().setRelationship(attacker.getTribeId(), target.getTribeId(), Types.RELATIONSHIP.WAR, gs.getTick());
        gs.getBoard().destroyEmbassiesBetween(gs, attacker.getTribeId(), target.getTribeId());

        boolean bomberAttack = attacker.getType() == Types.UNIT.BOMBER;
        Pair<Integer, Integer> results = getAttackResults(action, gs);
        int attackResult = results.getFirst();
        int defenceResult = results.getSecond();
        Vector2d targetPos = target.getPosition().copy();

        if (target.getCurrentHPExact() <= attackResult) {
            attacker.addKill();
            attackerTribe.addKill();
            gs.killUnit(target);

            if (bomberAttack) {
                applyBomberSplash(gs, attacker, targetPos);
            }

            if (isMeleePushUnit(attacker)) {
                gs.getBoard().tryPush(attackerTribe, attacker, attacker.getPosition().x, attacker.getPosition().y,
                        targetPos.x, targetPos.y, gs.getRandomGenerator());
            }
        } else {
            target.setCurrentHP(target.getCurrentHPExact() - attackResult);

            if (bomberAttack) {
                applyBomberSplash(gs, attacker, targetPos);
            }

            if (canRetaliate(action, attacker, target, gs)) {
                attacker.setCurrentHP(attacker.getCurrentHPExact() - defenceResult);
                if (attacker.getCurrentHPExact() <= 0) {
                    target.addKill();
                    gs.getTribe(target.getTribeId()).addKill();
                    gs.killUnit(attacker);
                }
            }
        }

        return true;
    }

    private void applyBomberSplash(GameState gs, Unit attacker, Vector2d targetPos) {
        for (Vector2d splashPos : targetPos.neighborhood(1, 0, gs.getBoard().getSize())) {
            Unit splashTarget = gs.getBoard().getUnitAt(splashPos.x, splashPos.y);
            if (splashTarget == null || splashTarget.getTribeId() == attacker.getTribeId()) {
                continue;
            }

            Attack splashAttack = new Attack(attacker.getActorId());
            splashAttack.setTargetId(splashTarget.getActorId());
            double splashDamage = getAttackResults(splashAttack, gs).getFirst() / 2.0;
            if (splashDamage <= 0) {
                continue;
            }
            if (splashTarget.getCurrentHPExact() <= splashDamage) {
                attacker.addKill();
                gs.getTribe(attacker.getTribeId()).addKill();
                gs.killUnit(splashTarget);
            } else {
                splashTarget.setCurrentHP(splashTarget.getCurrentHPExact() - splashDamage);
            }
        }
    }

    private boolean isMeleePushUnit(Unit attacker) {
        switch (attacker.getType()) {
            case DEFENDER:
            case SWORDMAN:
            case RIDER:
            case WARRIOR:
            case DAGGER:
            case KNIGHT:
            case SUPERUNIT:
            case RAMMER:
            case JUGGERNAUT:
            case PIRATE:
                return true;
            default:
                return false;
        }
    }

    private boolean canRetaliate(Attack action, Unit attacker, Unit target, GameState gs) {
        if (attacker.getType() == Types.UNIT.DAGGER || attacker.getType() == Types.UNIT.PIRATE) {
            return false;
        }

        double distance = Vector2d.chebychevDistance(attacker.getPosition(), target.getPosition());
        if (distance > target.RANGE) {
            return false;
        }

        if (target.getDefenceValue() <= 0 || hasStiffSkill(target)) {
            return false;
        }

        return true;
    }

    private boolean hasStiffSkill(Unit unit) {
        switch (unit.getType()) {
            case RAFT:
            case BOMBER:
            case CATAPULT:
            case MIND_BENDER:
            case CLOAK:
            case DINGHY:
            case JUGGERNAUT:
                return true;
            default:
                return false;
        }
    }

    /**
     * Calculates the damage dealt by the attacker and by the defender.
     *
     * @param gs - current game state
     * @return Pair, where first element is the attack power (attackResult) and second is defence power (defenceResult)
     */
    private Pair<Integer, Integer> getAttackResults(Attack action, GameState gs) {
        Unit attacker = (Unit) gs.getActor(action.getUnitId());
        Unit target = (Unit) gs.getActor(action.getTargetId());
        Vector2d targetPos = target.getPosition();
        Tribe targetTribe = gs.getTribe(target.getTribeId());

        double attackForce = attacker.getAttackValue() * (attacker.getCurrentHPExact() / attacker.getMaxHP());
        double defenceForce = target.getDefenceValue() * (target.getCurrentHPExact() / target.getMaxHP());
        double accelerator = TribesConfig.ATTACK_MODIFIER;

        Types.TERRAIN targetTerrain = gs.getBoard().getTerrainAt(targetPos.x, targetPos.y);
        if (targetTerrain == CITY) {
            int cityID = gs.getBoard().getCityIdAt(targetPos.x, targetPos.y);
            if (targetTribe.controlsCity(cityID) && target.getType().canFortify()) {
                City c = (City) gs.getActor(cityID);
                if (c.hasWalls()) {
                    defenceForce *= TribesConfig.DEFENCE_IN_WALLS;
                } else {
                    defenceForce *= TribesConfig.DEFENCE_BONUS;
                }
            }
        } else if ((targetTerrain == MOUNTAIN && targetTribe.getTechTree().isResearched(CLIMBING)) ||
                (targetTerrain.isWater() && targetTribe.getTechTree().isResearched(AQUATISM)) ||
                (targetTerrain == FOREST && targetTribe.getTechTree().isResearched(ARCHERY))) {
            defenceForce *= TribesConfig.DEFENCE_BONUS;
        }

        double totalDamage = attackForce + defenceForce;

        int attackResult = (int) Math.round((attackForce / totalDamage) * attacker.getAttackValue() * accelerator);
        int defenceResult = (int) Math.round((defenceForce / totalDamage) * target.getDefenceValue() * accelerator);

        return new Pair<>(attackResult, defenceResult);
    }

    public boolean isRetaliation(Attack action, GameState gs) {
        if (!action.isFeasible(gs)) {
            return false;
        }

        Unit attacker = (Unit) gs.getActor(action.getUnitId());
        Unit target = (Unit) gs.getActor(action.getTargetId());
        if (attacker == null || target == null) {
            return false;
        }

        int attackResult = getAttackResults(action, gs).getFirst();
        return target.getCurrentHPExact() > attackResult && canRetaliate(action, attacker, target, gs);
    }
}

