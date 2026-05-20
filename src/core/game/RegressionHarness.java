package core.game;

import core.Diplomacy;
import core.TechnologyTree;
import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.Spawn;
import core.actions.cityactions.command.BuildCommand;
import core.actions.cityactions.command.LevelUpCommand;
import core.actions.tribeactions.AcceptPeace;
import core.actions.tribeactions.AcceptTreaty;
import core.actions.tribeactions.BuildEmbassy;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.CancelTreaty;
import core.actions.tribeactions.ProposePeace;
import core.actions.tribeactions.ProposeTreaty;
import core.actions.tribeactions.command.AcceptPeaceCommand;
import core.actions.tribeactions.command.AcceptTreatyCommand;
import core.actions.tribeactions.command.BuildEmbassyCommand;
import core.actions.tribeactions.command.BuildRoadCommand;
import core.actions.tribeactions.command.CancelTreatyCommand;
import core.actions.tribeactions.command.ProposePeaceCommand;
import core.actions.tribeactions.command.ProposeTreatyCommand;
import core.actions.tribeactions.command.ResearchTechCommand;
import core.actions.tribeactions.ResearchTech;
import core.actions.unitactions.Convert;
import core.actions.unitactions.Disband;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Capture;
import core.actions.unitactions.Examine;
import core.actions.unitactions.Infiltrate;
import core.actions.unitactions.MakeVeteran;
import core.actions.unitactions.Move;
import core.actions.unitactions.Recover;
import core.actions.unitactions.command.AttackCommand;
import core.actions.unitactions.command.CaptureCommand;
import core.actions.unitactions.command.DisbandCommand;
import core.actions.unitactions.command.ExamineCommand;
import core.actions.unitactions.command.InfiltrateCommand;
import core.actions.unitactions.command.MakeVeteranCommand;
import core.actions.unitactions.command.MoveCommand;
import core.actions.unitactions.command.RecoverCommand;
import core.actions.unitactions.factory.MoveFactory;
import core.actors.Building;
import core.actors.City;
import core.actors.Temple;
import core.actors.Tribe;
import core.actors.units.CarrierUnit;
import core.actors.units.Unit;
import core.levelgen.LevelGenerator;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.external.ExternalBotPayloadBuilder;
import players.external.ExternalForwardModelSession;
import utils.ElapsedCpuTimer;
import utils.Vector2d;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.stream.Stream;

public final class RegressionHarness {

    private static final long SCENARIO_SEED = 1337L;
    private static final long SLOW_TEST_MS = 250L;

    private RegressionHarness() {
    }

    public static void main(String[] args) {
        String suiteFilter = parseFlagValue(args, "--suite");
        String testFilter = parseFlagValue(args, "--test");
        HarnessSuite[] suites = new HarnessSuite[]{
                suite("Technology and Modes",
                        test("tech cost scaling", RegressionHarness::testTechCostScaling),
                        test("technology tree stable-id serialization", RegressionHarness::testTechnologyTreeStableIdSerialization),
                        test("technology tree legacy-array loading", RegressionHarness::testTechnologyTreeLegacyArrayLoading),
                        test("strategy and diplomacy research path", RegressionHarness::testStrategyAndDiplomacyResearchPath),
                        test("technology tree completion tracks diplomacy leaf", RegressionHarness::testTechnologyTreeCompletionTracksDiplomacyLeaf),
                        test("fishing unlocks ports without sailing", RegressionHarness::testFishingUnlocksPortWithoutSailing),
                        test("live map sizes keep supported side lengths", RegressionHarness::testLiveMapSizesMatchSupportedSideLengths),
                        test("game mode aliases map to live canonical modes", RegressionHarness::testGameModeCanonicalModes),
                        test("perfection ends on the turn limit", RegressionHarness::testPerfectionEndsOnTurnLimit),
                        test("turn-limit override draws stalled might games", RegressionHarness::testTurnLimitOverrideDrawsMightGames),
                        test("glory resolves at round end when the target is reached", RegressionHarness::testGloryResolvesAtRoundEnd),
                        test("domination ends when one tribe remains", RegressionHarness::testDominationEndsWhenOneTribeRemains)
                ),
                suite("Diplomacy and Visibility",
                        test("diplomacy research reveals met capitals", RegressionHarness::testDiplomacyResearchRevealsMetCapitals),
                        test("meeting after diplomacy reveals capital", RegressionHarness::testMeetingAfterDiplomacyRevealsCapital),
                        test("meeting star reward is capped at eleven", RegressionHarness::testMeetingStarRewardIsCappedAtEleven),
                        test("known capital query follows visibility knowledge", RegressionHarness::testKnownCapitalQueryFollowsVisibilityKnowledge),
                        test("embassy build action and placement rules", RegressionHarness::testEmbassyBuildActionAndPlacementRules),
                        test("embassy income at peace", RegressionHarness::testEmbassyIncomeAtPeace),
                        test("embassy income scales with treaty", RegressionHarness::testEmbassyIncomeScalesWithTreaty),
                        test("hostile attack destroys embassies and starts war", RegressionHarness::testHostileAttackDestroysEmbassiesAndStartsWar),
                        test("captured capitals cannot keep embassies", RegressionHarness::testCapturedCapitalsCannotKeepEmbassies),
                        test("relationship visibility without diplomacy is local only", RegressionHarness::testRelationshipVisibilityWithoutDiplomacyIsLocalOnly),
                        test("relationship visibility with diplomacy reveals all tribes", RegressionHarness::testRelationshipVisibilityWithDiplomacyRevealsAllTribes),
                        test("diplomacy relations default to peace", RegressionHarness::testDiplomacyRelationsDefaultToPeace),
                        test("peace offers can end wars", RegressionHarness::testPeaceOffersCanEndWars),
                        test("treaty offers can be accepted and canceled", RegressionHarness::testTreatyOffersCanBeAcceptedAndCanceled),
                        test("cancel treaty keeps embassies and disbands border intruders", RegressionHarness::testCancelTreatyKeepsEmbassiesAndDisbandsIntruders),
                        test("cancel treaty finishes remaining units for the turn", RegressionHarness::testCancelTreatyFinishesRemainingUnits),
                        test("treaty blocks hostile legal actions", RegressionHarness::testTreatyBlocksHostileLegalActions),
                        test("treaty allies can use allied roads", RegressionHarness::testTreatyAlliesCanUseAlliedRoads),
                        test("generated legal actions stay feasible in diplomacy scenario", RegressionHarness::testGeneratedLegalActionsStayFeasibleInDiplomacyScenario),
                        test("diplomacy unlocks cloak spawning", RegressionHarness::testDiplomacyUnlocksCloakSpawning),
                        test("hidden cloaks are redacted from observation copies", RegressionHarness::testHiddenCloakRedactedFromObservationCopy),
                        test("adjacent units do not reveal hidden cloaks", RegressionHarness::testAdjacentUnitsDoNotRevealHiddenCloaks),
                        test("freshly spawned cloaks stay visible until they move", RegressionHarness::testFreshlySpawnedCloaksStayVisibleUntilMove),
                        test("explored tiles persist without visibility refresh", RegressionHarness::testExploredTilesPersistWithoutVisibilityRefresh),
                        test("visibility map uses exploration state", RegressionHarness::testVisibilityMapUsesExplorationState),
                        test("scout and cloak use extended reveal", RegressionHarness::testScoutAndCloakRevealRange),
                        test("mountains extend normal reveal by one tile", RegressionHarness::testMountainRevealRange),
                        test("move generation requires explored destination", RegressionHarness::testMoveGenerationRequiresExploredDestination),
                        test("observation copies do not invent enemy road access", RegressionHarness::testObservationCopiesDoNotInventEnemyRoadAccess),
                        test("moving into a hidden cloak reveals and blocks movement", RegressionHarness::testMovingIntoHiddenCloakRevealsAndBlocksMovement),
                        test("move command rejects occupied destination", RegressionHarness::testMoveCommandRejectsOccupiedDestination),
                        test("scout disembark keeps scout reveal range", RegressionHarness::testScoutDisembarkKeepsScoutRevealRange),
                        test("lighthouse discovery rewards each lighthouse once", RegressionHarness::testLighthouseDiscoveryRewardsEachLighthouseOnce)
                ),
                suite("Economy and Infrastructure",
                        test("market income matches adjacent support building levels", RegressionHarness::testMarketIncomeMatchesSupportBuildingLevels),
                        test("market requires adjacent friendly support building to be built", RegressionHarness::testMarketBuildRequiresAdjacentFriendlySupport),
                        test("roads can be built on explored fogged tiles", RegressionHarness::testRoadsCanBeBuiltOnExploredFoggedTiles),
                        test("bridge builds on explored neutral water for five stars", RegressionHarness::testBridgeBuildsOnExploredNeutralWaterForFiveStars),
                        test("bridge can target fogged far shoreline", RegressionHarness::testBridgeCanTargetFoggedFarShoreline),
                        test("build actions require explored empty city tiles", RegressionHarness::testBuildActionsRequireExploredEmptyCityTiles),
                        test("connecting roads add population to capital and city", RegressionHarness::testConnectingRoadsAddPopulationToCapitalAndCity),
                        test("unexplored road segments break city connections", RegressionHarness::testUnexploredRoadSegmentsBreakCityConnections),
                        test("allied roads can complete city connections", RegressionHarness::testAlliedRoadsCanCompleteCityConnections),
                        test("ocean port links require navigation", RegressionHarness::testOceanPortLinksRequireNavigation),
                        test("port links span five water tiles", RegressionHarness::testPortLinksSpanFiveWaterTiles),
                        test("discovering lighthouses unlocks eye of god", RegressionHarness::testLighthouseDiscoveryUnlocksEyeOfGod),
                        test("border growth reveals expanded border lighthouses", RegressionHarness::testBorderGrowthRevealsExpandedBorderLighthouses),
                        test("visible lighthouses show discoverers", RegressionHarness::testVisibleLighthousesShowDiscoverers),
                        test("city level-up behavior", RegressionHarness::testCityLevelUpBehavior),
                        test("city level-up points keep declining after level five", RegressionHarness::testCityLevelUpPointsDeclineAfterLevelFive),
                        test("negative population reduces city production without making it negative", RegressionHarness::testNegativePopulationReducesCityProduction),
                        test("recover gains territory bonus off city tile", RegressionHarness::testRecoverGetsTerritoryBonusInBorders)
                ),
                suite("Combat and Ruins",
                        test("unit stats match current wiki values", RegressionHarness::testUnitStatsMatchWiki),
                        test("cloak cannot become veteran", RegressionHarness::testCloakCannotBecomeVeteran),
                        test("disband refunds half cost and naval units refund only carried unit cost", RegressionHarness::testDisbandRefundsMatchWiki),
                        test("infiltration spawns daggers and starts war", RegressionHarness::testInfiltrationSpawnsDaggersAndStartsWar),
                        test("dagger attacks skip retaliation", RegressionHarness::testDaggerAttacksSkipRetaliation),
                        test("retaliation ignores attacker visibility once attack is legal", RegressionHarness::testRetaliationIgnoresAttackerVisibility),
                        test("ruin stars always grant ten stars", RegressionHarness::testRuinStarsAlwaysGrantTenStars),
                        test("ruin population grants three capital population", RegressionHarness::testRuinPopulationGivesThreeCapitalPopulation),
                        test("ruin research grants exactly one new technology", RegressionHarness::testRuinResearchGrantsExactlyOneTechnology),
                        test("water ruins can spawn veteran rammer and push the examiner aside", RegressionHarness::testWaterRuinSpawnsVeteranRammer),
                        test("temples score one hundred per level", RegressionHarness::testTempleScoringMatchesWiki),
                        test("temples and monuments update city points when built or destroyed", RegressionHarness::testBuildingScoreUpdatesCityPointsWorth),
                        test("combat retaliation", RegressionHarness::testCombatRetaliation),
                        test("splash damage preserves half hit points", RegressionHarness::testSplashDamagePreservesHalfHitPoints),
                        test("city capture", RegressionHarness::testCityCapture)
                ),
                suite("Map Generation",
                        test("level generator rejects unsupported map sizes", RegressionHarness::testLevelGeneratorRejectsUnsupportedMapSizes),
                        test("level generator keeps confirmed capital guarantees", RegressionHarness::testLevelGeneratorKeepsConfirmedCapitalGuarantees),
                        test("level generator avoids shallow-water ruins", RegressionHarness::testLevelGeneratorAvoidsShallowWaterRuins),
                        test("ocean ruins stay at least two tiles from villages", RegressionHarness::testOceanRuinsStayTwoTilesFromVillages),
                        test("deep water never touches land orthogonally", RegressionHarness::testDeepWaterNeverTouchesLandOrthogonally),
                        test("map types follow expected water profiles", RegressionHarness::testMapTypesFollowExpectedWaterProfiles),
                        test("lakes capitals keep land access to two villages", RegressionHarness::testLakesCapitalsKeepLandAccessToTwoVillages),
                        test("lakes villages keep spacing from capitals and each other", RegressionHarness::testLakesVillageSpacingFromCapitalsAndEachOther),
                        test("continents place capitals on separate landmasses", RegressionHarness::testContinentsPlaceCapitalsOnSeparateLandmasses),
                        test("continents limit one tile island villages", RegressionHarness::testContinentsLimitOneTileIslandVillages),
                        test("continents allow edge land without walkable continent contact", RegressionHarness::testContinentsAllowEdgeLandWithoutWalkableContinentContact),
                        test("continents keep capitals spaced and give each landmass villages", RegressionHarness::testContinentsKeepCapitalsSpacedAndGiveEachLandmassVillages),
                        test("generated land resources stay near cities", RegressionHarness::testGeneratedLandResourcesStayNearCities),
                        test("generated ruins obey terrain and spacing rules", RegressionHarness::testGeneratedRuinsObeyTerrainAndSpacing),
                        test("starfish density follows water tile count", RegressionHarness::testStarfishDensityFollowsWaterTileCount),
                        test("level generator places corner lighthouses", RegressionHarness::testLevelGeneratorPlacesCornerLighthouses),
                        test("level generator uses live ruin counts", RegressionHarness::testLevelGeneratorUsesLiveRuinCounts),
                        test("level generator keeps ruins off lighthouse corners", RegressionHarness::testLevelGeneratorKeepsRuinsOffLighthouseCorners),
                        test("level generator spaces starfish away from cities lighthouses and starfish", RegressionHarness::testLevelGeneratorSpacesStarfish)
                ),
                suite("Persistence and Bots",
                        test("hidden cloak state survives save load", RegressionHarness::testHiddenCloakStateSurvivesSaveLoad),
                        test("diplomacy state survives save load", RegressionHarness::testDiplomacyStateSurvivesSaveLoad),
                        test("game state copy clones RNG state", RegressionHarness::testGameStateCopyClonesRandomState),
                        test("tribe result tie break is deterministic", RegressionHarness::testTribeResultTieBreakIsDeterministic),
                        test("duplicate tribe seats preserve player assignment", RegressionHarness::testDuplicateTribeSeatsPreservePlayerAssignment),
                        test("external action request uses compact schema", RegressionHarness::testExternalActionRequestUsesCompactSchema),
                        test("external payload includes hidden hints and lighthouse discoverers", RegressionHarness::testExternalPayloadIncludesHiddenHintsAndLighthouseDiscoverers),
                        test("external forward model state uses compact schema", RegressionHarness::testExternalForwardModelStateUsesCompactSchema),
                        test("external game over uses compact schema", RegressionHarness::testExternalGameOverUsesCompactSchema),
                        test("save/load round-trip", RegressionHarness::testSaveLoadRoundTrip),
                        test("turn-sensitive save/load state survives round-trip", RegressionHarness::testTurnSensitiveSaveLoadStateSurvivesRoundTrip)
                )
        };

        int passed = 0;
        int total = 0;
        long startNanos = System.nanoTime();

        for (HarnessSuite suite : suites) {
            if (suiteFilter != null && !suite.name.toLowerCase().contains(suiteFilter.toLowerCase())) {
                continue;
            }
            int suiteTotal = countMatchingTests(suite, testFilter);
            if (suiteTotal == 0) {
                continue;
            }
            total += suiteTotal;
            passed += runSuite(suite, testFilter);
        }

        if (passed != total) {
            System.err.println("Regression harness failed " + (total - passed) + " of " + total + " checks.");
            System.exit(1);
        }

        long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        System.out.println("Regression harness passed " + passed + " of " + total + " checks in " + elapsedMs + " ms.");
    }

    private static HarnessSuite suite(String name, HarnessTest... tests) {
        return new HarnessSuite(name, tests);
    }

    private static HarnessTest test(String name, CheckedRunnable check) {
        return new HarnessTest(name, check);
    }

    private static int runSuite(HarnessSuite suite, String testFilter) {
        System.out.println("[SUITE] " + suite.name);
        int passed = 0;
        int runCount = 0;
        for (HarnessTest test : suite.tests) {
            if (testFilter != null && !test.name.toLowerCase().contains(testFilter.toLowerCase())) {
                continue;
            }
            passed += runCheck(test.name, test.check);
            runCount++;
        }
        System.out.println("[SUITE PASS] " + suite.name + " " + passed + "/" + runCount);
        return passed;
    }

    private static int runCheck(String name, CheckedRunnable check) {
        long startNanos = System.nanoTime();
        try {
            check.run();
            long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
            System.out.println("[PASS] " + name + formatTiming(elapsedMs));
            return 1;
        } catch (Throwable t) {
            long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
            System.err.println("[FAIL] " + name + ": " + t.getMessage());
            if (elapsedMs >= SLOW_TEST_MS) {
                System.err.println("[SLOW] " + name + " took " + elapsedMs + " ms before failing.");
            }
            t.printStackTrace(System.err);
            return 0;
        }
    }

    private static void testTechCostScaling() {
        TechnologyTree tree = new TechnologyTree();

        assertEquals(5, Types.TECHNOLOGY.CLIMBING.getCost(1, tree), "Tier-1 tech cost should start at base + cities");
        assertEquals(7, Types.TECHNOLOGY.CLIMBING.getCost(3, tree), "Tier-1 tech cost should scale with city count");
        assertEquals(10, Types.TECHNOLOGY.SMITHERY.getCost(2, tree), "Tier-3 tech cost should scale with both tier and city count");

        tree.doResearchInit(Types.TECHNOLOGY.PHILOSOPHY);
        assertEquals(8, Types.TECHNOLOGY.CLIMBING.getCost(7, tree), "Philosophy discount should reduce tech costs by one third, rounded up");
        assertEquals(11, Types.TECHNOLOGY.TRADE.getCost(4, tree), "Discounted higher-tier tech cost should reduce costs by one third, rounded up");
    }

    private static void testTechnologyTreeStableIdSerialization() {
        TechnologyTree original = new TechnologyTree();
        original.doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        original.doResearchInit(Types.TECHNOLOGY.FARMING);
        original.doResearchInit(Types.TECHNOLOGY.STRATEGY);

        JSONObject serialized = new JSONObject();
        serialized.put("researchedTechIds", original.getResearchedTechIds());
        serialized.put("everythingResearched", original.isEverythingResearched());

        TechnologyTree loaded = new TechnologyTree(serialized);
        assertTrue(loaded.isResearched(Types.TECHNOLOGY.ORGANIZATION), "Stable-id load should preserve Organization");
        assertTrue(loaded.isResearched(Types.TECHNOLOGY.FARMING), "Stable-id load should preserve Farming");
        assertTrue(loaded.isResearched(Types.TECHNOLOGY.STRATEGY), "Stable-id load should preserve Strategy");
        assertFalse(loaded.isResearched(Types.TECHNOLOGY.ARCHERY), "Stable-id load should not add unrelated techs");
    }

    private static void testTechnologyTreeLegacyArrayLoading() {
        JSONObject legacy = new JSONObject();
        JSONArray researched = new JSONArray();
        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            researched.put(technology == Types.TECHNOLOGY.CLIMBING || technology == Types.TECHNOLOGY.MINING);
        }
        legacy.put("researched", researched);
        legacy.put("everythingResearched", false);

        TechnologyTree loaded = new TechnologyTree(legacy);
        assertTrue(loaded.isResearched(Types.TECHNOLOGY.CLIMBING), "Legacy array load should preserve Climbing");
        assertTrue(loaded.isResearched(Types.TECHNOLOGY.MINING), "Legacy array load should preserve Mining");
        assertFalse(loaded.isResearched(Types.TECHNOLOGY.ROADS), "Legacy array load should not add unrelated techs");
    }

    private static void testStrategyAndDiplomacyResearchPath() {
        assertEquals(Types.TECHNOLOGY.ORGANIZATION, Types.TECHNOLOGY.STRATEGY.getParentTech(),
                "Strategy should sit behind Organization.");
        assertEquals(Types.TECHNOLOGY.STRATEGY, Types.TECHNOLOGY.DIPLOMACY.getParentTech(),
                "Diplomacy should sit behind Strategy.");
        assertEquals(10, Types.TECHNOLOGY.DIPLOMACY.getCost(2, new TechnologyTree()),
                "Diplomacy should use the configured tier-3 tech cost formula.");

        Scenario strategyScenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        strategyScenario.addCapital(0, 2, 2);
        strategyScenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        strategyScenario.tribes[0].setStars(20);
        GameState strategyState = strategyScenario.buildState();
        strategyState.computePlayerActions(strategyState.getActiveTribe());

        assertTrue(hasResearchAction(strategyState, Types.TECHNOLOGY.STRATEGY),
                "Strategy should be a legal research action once Organization is researched.");
        assertFalse(hasResearchAction(strategyState, Types.TECHNOLOGY.DIPLOMACY),
                "Diplomacy should not be legal before Strategy is researched.");

        Scenario diplomacyScenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        diplomacyScenario.addCapital(0, 2, 2);
        diplomacyScenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        diplomacyScenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        diplomacyScenario.tribes[0].setStars(20);
        GameState diplomacyState = diplomacyScenario.buildState();
        diplomacyState.computePlayerActions(diplomacyState.getActiveTribe());

        assertTrue(hasResearchAction(diplomacyState, Types.TECHNOLOGY.DIPLOMACY),
                "Diplomacy should become a legal research action once Strategy is researched.");
    }

    private static void testTechnologyTreeCompletionTracksDiplomacyLeaf() {
        TechnologyTree tree = new TechnologyTree();
        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            if (technology != Types.TECHNOLOGY.DIPLOMACY) {
                tree.doResearchInit(technology);
            }
        }

        assertFalse(tree.isEverythingResearched(),
                "Everything researched should stay false while Diplomacy remains unresearched.");
        assertTrue(tree.doResearch(Types.TECHNOLOGY.DIPLOMACY),
                "Diplomacy should still be researchable after its prerequisite path is marked.");
        assertTrue(tree.isEverythingResearched(),
                "Researching the new leaf should mark the whole tree complete.");
    }

    private static void testFishingUnlocksPortWithoutSailing() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.tribes[0].setStars(20);

        Build withoutFishing = new Build(capital.getActorId());
        withoutFishing.setBuildingType(Types.BUILDING.PORT);
        withoutFishing.setTargetPos(new Vector2d(2, 3));

        GameState noFishingState = scenario.buildState();
        assertFalse(withoutFishing.isFeasible(noFishingState),
                "Ports should not be buildable before Fishing is researched.");

        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FISHING);
        GameState fishingState = scenario.buildState();
        Build withFishing = new Build(capital.getActorId());
        withFishing.setBuildingType(Types.BUILDING.PORT);
        withFishing.setTargetPos(new Vector2d(2, 3));

        assertFalse(scenario.tribes[0].getTechTree().isResearched(Types.TECHNOLOGY.SAILING),
                "This regression specifically checks that Sailing is not required for ports.");
        assertTrue(withFishing.isFeasible(fishingState),
                "Fishing alone should unlock Port construction.");
        assertTrue(new BuildCommand().execute(withFishing, fishingState),
                "Port build should execute once Fishing is researched.");
        assertEquals(Types.BUILDING.PORT, fishingState.getBoard().getBuildingAt(2, 3),
                "Executing the Port build should place a port on the water tile.");
    }

    private static void testLiveMapSizesMatchSupportedSideLengths() {
        assertEquals(11, Types.MAP_SIZE.TINY.getSideLength(), "Tiny maps should stay 11x11.");
        assertEquals(14, Types.MAP_SIZE.SMALL.getSideLength(), "Small maps should stay 14x14.");
        assertEquals(16, Types.MAP_SIZE.NORMAL.getSideLength(), "Normal maps should stay 16x16.");
        assertEquals(18, Types.MAP_SIZE.LARGE.getSideLength(), "Large maps should stay 18x18.");
        assertEquals(20, Types.MAP_SIZE.HUGE.getSideLength(), "Huge maps should stay 20x20.");
        assertEquals(30, Types.MAP_SIZE.MASSIVE.getSideLength(), "Massive maps should stay 30x30.");
    }

    private static void testDiplomacyResearchRevealsMetCapitals() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        scenario.tribes[0].setStars(20);

        GameState gs = scenario.buildState();
        Vector2d enemyCapitalPos = enemyCapital.getPosition();
        assertFalse(gs.getKnownCapitalPositions(0).containsKey(1),
                "Met tribes should not reveal capitals before Diplomacy is researched.");

        ResearchTech research = new ResearchTech(0);
        research.setTech(Types.TECHNOLOGY.DIPLOMACY);
        assertTrue(research.isFeasible(gs), "Diplomacy research should be feasible once prerequisites are met.");
        assertTrue(new ResearchTechCommand().execute(research, gs), "Diplomacy research should execute.");

        assertTrue(gs.getTribe(0).getKnownCapitalTribes().contains(1),
                "Researching Diplomacy should reveal capitals of tribes already met.");

        HashMap<Integer, Vector2d> knownCapitals = gs.getKnownCapitalPositions(0);
        assertVectorEquals(enemyCapitalPos, knownCapitals.get(1),
                "Known-capital query should expose capitals revealed by Diplomacy.");
    }

    private static void testMeetingAfterDiplomacyRevealsCapital() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.addUnit(enemyCapital, Types.UNIT.WARRIOR, 4, 1);

        GameState gs = scenario.buildState();
        Tribe observer = gs.getTribe(0);
        Vector2d enemyCapitalPos = enemyCapital.getPosition();

        assertFalse(observer.getTribesMet().contains(1),
                "Scenario should begin before the tribes have met.");
        assertFalse(gs.getKnownCapitalPositions(0).containsKey(1),
                "Enemy capital should not be known before first contact.");

        observer.clearView(4, 1, 0, gs.getRandomGenerator(), gs.getBoard());

        assertTrue(observer.getTribesMet().contains(1),
                "Revealing an enemy unit should mark the tribe as met.");
        assertTrue(gs.getKnownCapitalPositions(0).containsKey(1),
                "Meeting a tribe after researching Diplomacy should reveal its capital.");
    }

    private static void testMeetingStarRewardIsCappedAtEleven() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        scenario.tribes[1].setScore(9000);

        GameState gs = scenario.buildState();
        int starsBefore = scenario.tribes[0].getStars();
        scenario.tribes[0].clearView(enemyCapital.getPosition().x, enemyCapital.getPosition().y, 0,
                gs.getRandomGenerator(), gs.getBoard());

        assertEquals(starsBefore + 11, scenario.tribes[0].getStars(),
                "Meeting a high-score tribe should cap the contact reward at eleven stars.");
    }

    private static void testKnownCapitalQueryFollowsVisibilityKnowledge() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        GameState gs = scenario.buildState();

        HashMap<Integer, Vector2d> knownCapitals = gs.getKnownCapitalPositions(0);
        assertVectorEquals(homeCapital.getPosition(), knownCapitals.get(0),
                "A tribe should always know its own capital.");
        assertFalse(knownCapitals.containsKey(1),
                "Enemy capitals should not be queryable before they become visible.");

        gs.getTribe(0).revealCapitalOfTribe(gs.getBoard(), 1);
        knownCapitals = gs.getKnownCapitalPositions(0);
        assertVectorEquals(enemyCapital.getPosition(), knownCapitals.get(1),
                "Known-capital query should include enemy capitals once their tile is visible.");
    }

    private static void testEmbassyBuildActionAndPlacementRules() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].setStars(10);
        assertFalse(scenario.board.canBuildEmbassy(0, 1),
                "Embassies should not be buildable before the tribe has made contact.");

        scenario.tribes[0].getTribesMet().add(1);

        GameState gs = scenario.buildState();
        gs.computePlayerActions(gs.getActiveTribe());
        assertTrue(hasTribeActionType(gs, Types.ACTION.BUILD_EMBASSY),
                "Diplomacy and contact should unlock embassy placement.");

        BuildEmbassy buildEmbassy = new BuildEmbassy(0);
        buildEmbassy.setTargetID(1);
        assertTrue(buildEmbassy.isFeasible(gs), "Embassy build should be feasible in a met tribe's capital.");
        assertTrue(new BuildEmbassyCommand().execute(buildEmbassy, gs), "Embassy build should execute.");
        assertEquals(5, scenario.tribes[0].getStars(), "Embassy build should cost 5 stars.");
        assertTrue(gs.getBoard().hasEmbassy(0, 1), "Embassy should be registered in the target capital.");

        gs.getBoard().capture(gs, scenario.tribes[0], enemyCapital.getPosition().x, enemyCapital.getPosition().y);
        assertFalse(gs.getBoard().canBuildEmbassy(0, 1),
                "Embassies should not be buildable in a captured capital.");
        assertEquals(homeCapital.getActorId(), scenario.tribes[0].getCapitalID(),
                "Building an embassy must not change the builder's own capital state.");
    }

    private static void testEmbassyIncomeAtPeace() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[0].setStars(10);

        GameState gs = scenario.buildState();
        BuildEmbassy buildEmbassy = new BuildEmbassy(0);
        buildEmbassy.setTargetID(1);
        assertTrue(new BuildEmbassyCommand().execute(buildEmbassy, gs), "Embassy build should execute before income is tested.");
        assertEquals(TribesConfig.EMBASSY_STARS, gs.getBoard().getEmbassyIncomeForTribe(0),
                "Embassy owner should receive the peace-time income amount.");
        assertEquals(TribesConfig.EMBASSY_STARS, gs.getBoard().getEmbassyIncomeForTribe(1),
                "Embassy host should also receive the peace-time income amount.");

        int tribe0BaseIncome = scenario.tribes[0].getMaxProduction(gs);
        int tribe1BaseIncome = scenario.tribes[1].getMaxProduction(gs);
        gs.incTick();
        scenario.tribes[0].setStars(0);
        scenario.tribes[1].setStars(0);
        gs.initTurn(scenario.tribes[0]);
        gs.initTurn(scenario.tribes[1]);

        assertEquals(tribe0BaseIncome + TribesConfig.EMBASSY_STARS, scenario.tribes[0].getStars(),
                "Embassy owner should receive recurring embassy stars on turn start.");
        assertEquals(tribe1BaseIncome + TribesConfig.EMBASSY_STARS, scenario.tribes[1].getStars(),
                "Embassy host should receive recurring embassy stars on turn start.");
    }

    private static void testEmbassyIncomeScalesWithTreaty() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[0].setStars(10);

        GameState gs = scenario.buildState();
        BuildEmbassy buildEmbassy = new BuildEmbassy(0);
        buildEmbassy.setTargetID(1);
        assertTrue(new BuildEmbassyCommand().execute(buildEmbassy, gs), "Embassy build should execute before treaty income is tested.");
        gs.getTribe(0).revealArea(gs.getBoard(), 3, 2, 0);
        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 2);

        int tribe0BaseIncome = scenario.tribes[0].getMaxProduction(gs);
        int tribe1BaseIncome = scenario.tribes[1].getMaxProduction(gs);
        gs.incTick();
        scenario.tribes[0].setStars(0);
        scenario.tribes[1].setStars(0);
        gs.initTurn(scenario.tribes[0]);
        gs.initTurn(scenario.tribes[1]);

        assertEquals(tribe0BaseIncome + TribesConfig.EMBASSY_TREATY_STARS, scenario.tribes[0].getStars(),
                "Treaty allies should receive the boosted embassy income.");
        assertEquals(tribe1BaseIncome + TribesConfig.EMBASSY_TREATY_STARS, scenario.tribes[1].getStars(),
                "Treaty hosts should also receive the boosted embassy income.");
    }

    private static void testHostileAttackDestroysEmbassiesAndStartsWar() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = scenario.addCapital(0, 0, 0);
        City defenderCapital = scenario.addCapital(1, 4, 4);
        Unit attacker = scenario.addUnit(attackerCapital, Types.UNIT.WARRIOR, 2, 2);
        Unit defender = scenario.addUnit(defenderCapital, Types.UNIT.WARRIOR, 2, 3);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[1].getTribesMet().add(0);
        scenario.tribes[0].setStars(10);
        scenario.tribes[1].setStars(10);

        GameState gs = scenario.buildState();
        BuildEmbassy embassy01 = new BuildEmbassy(0);
        embassy01.setTargetID(1);
        BuildEmbassy embassy10 = new BuildEmbassy(1);
        embassy10.setTargetID(0);
        assertTrue(new BuildEmbassyCommand().execute(embassy01, gs), "Embassy 0->1 should build before the hostile attack.");
        assertTrue(new BuildEmbassyCommand().execute(embassy10, gs), "Embassy 1->0 should build before the hostile attack.");

        attacker.setStatus(Types.TURN_STATUS.FRESH);
        defender.setStatus(Types.TURN_STATUS.FRESH);
        Attack attack = new Attack(attacker.getActorId());
        attack.setTargetId(defender.getActorId());
        assertTrue(new AttackCommand().execute(attack, gs), "Neutral attack should execute.");

        assertEquals(Types.RELATIONSHIP.WAR, gs.getBoard().getDiplomacy().getRelationship(0, 1),
                "A hostile attack should switch the relationship to war.");
        assertFalse(gs.getBoard().hasEmbassy(0, 1), "Hostile attacks should destroy the attacker's embassy in the target capital.");
        assertFalse(gs.getBoard().hasEmbassy(1, 0), "Hostile attacks should destroy the defender's embassy in the attacker's capital.");
    }

    private static void testCapturedCapitalsCannotKeepEmbassies() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = scenario.addCapital(0, 0, 0);
        City defenderCapital = scenario.addCapital(1, 2, 2);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[0].setStars(10);
        Unit attacker = scenario.addUnit(attackerCapital, Types.UNIT.WARRIOR, 2, 2);

        GameState gs = scenario.buildState();
        BuildEmbassy buildEmbassy = new BuildEmbassy(0);
        buildEmbassy.setTargetID(1);
        assertTrue(new BuildEmbassyCommand().execute(buildEmbassy, gs), "Embassy should exist before the capital is captured.");
        assertTrue(gs.getBoard().hasEmbassy(0, 1), "Embassy should be present before capital capture.");

        attacker.setStatus(Types.TURN_STATUS.FRESH);
        Capture capture = new Capture(attacker.getActorId());
        capture.setCaptureType(Types.TERRAIN.CITY);
        capture.setTargetCity(defenderCapital.getActorId());
        assertTrue(new CaptureCommand().execute(capture, gs), "Capital capture should execute.");

        assertFalse(gs.getBoard().hasEmbassy(0, 1), "Capturing a capital should destroy embassies in that capital.");
        assertFalse(gs.getBoard().canBuildEmbassy(0, 1),
                "Embassies should remain illegal while the original owner no longer controls the capital.");
    }

    private static void testRelationshipVisibilityWithoutDiplomacyIsLocalOnly() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.HOODRICK);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        scenario.addCapital(2, 2, 4);
        GameState gs = scenario.buildState();

        gs.getBoard().getDiplomacy().setRelationship(1, 2, Types.RELATIONSHIP.TREATY, 3);
        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.WAR, 3);

        Types.RELATIONSHIP[][] visible = gs.getVisibleRelationships(0);
        assertEquals(Types.RELATIONSHIP.WAR, visible[0][1],
                "Without Diplomacy, a tribe should still see its own direct relationships.");
        assertEquals(Types.RELATIONSHIP.WAR, visible[1][0],
                "Direct relationship visibility should remain symmetric for the querying tribe.");
        assertEquals(null, visible[1][2],
                "Without Diplomacy, third-party relationships should stay hidden.");
        assertEquals(null, visible[2][1],
                "Hidden third-party relationships should remain unknown in both directions.");
    }

    private static void testRelationshipVisibilityWithDiplomacyRevealsAllTribes() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.HOODRICK);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        scenario.addCapital(2, 2, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        GameState gs = scenario.buildState();

        gs.getBoard().getDiplomacy().setRelationship(1, 2, Types.RELATIONSHIP.TREATY, 3);
        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.WAR, 3);

        Types.RELATIONSHIP[][] visible = gs.getVisibleRelationships(0);
        assertEquals(Types.RELATIONSHIP.WAR, visible[0][1],
                "Diplomacy should still expose the querying tribe's direct relationships.");
        assertEquals(Types.RELATIONSHIP.TREATY, visible[1][2],
                "Diplomacy should reveal third-party relationships.");
        assertEquals(Types.RELATIONSHIP.TREATY, visible[2][1],
                "Third-party visibility should preserve symmetry.");
    }

    private static void testDiplomacyRelationsDefaultToPeace() {
        Diplomacy diplomacy = new Diplomacy(3);
        assertEquals(Types.RELATIONSHIP.PEACE, diplomacy.getRelationship(0, 1),
                "New diplomacy state should start at peace between tribes.");
        assertEquals(Types.RELATIONSHIP.PEACE, diplomacy.getRelationship(1, 2),
                "All tribe pairs should start at peace.");
        assertFalse(diplomacy.isAtWar(0, 1), "Fresh diplomacy state should not mark tribes as at war.");
    }

    private static void testDiplomacyStateSurvivesSaveLoad() throws IOException {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        GameState gs = scenario.buildState();
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        diplomacy.setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 4);
        diplomacy.setAllegianceStatus(0, 1, 12);
        diplomacy.setAllegianceStatus(1, 0, 12);
        diplomacy.setPendingTreatyOffer(0, 1, 0);
        diplomacy.setPendingOfferType(0, 1, Types.RELATIONSHIP.TREATY);
        scenario.tribes[0].revealCapitalOfTribe(gs.getBoard(), 1);

        long seed = System.nanoTime();
        Path saveRoot = Paths.get("save", Long.toString(seed));
        Path saveFile = saveRoot.resolve(gs.getTick() + "_" + gs.getActiveTribeID()).resolve("game.json");

        try {
            Files.createDirectories(saveRoot);
            GameSaver.writeTurnFile(gs, scenario.board, seed);

            GameLoader loader = new GameLoader(saveFile.toString());
            Diplomacy loadedDiplomacy = loader.getBoard().getDiplomacy();

            assertEquals(Types.RELATIONSHIP.TREATY, loadedDiplomacy.getRelationship(0, 1),
                    "Diplomacy relationship should survive save/load.");
            assertEquals(12, loadedDiplomacy.getAllegianceStatus()[0][1],
                    "Allegiance should survive save/load.");
            assertEquals(4, loadedDiplomacy.getLastRelationChangeTurn(0, 1),
                    "Relation-change metadata should survive save/load.");
            assertEquals(0, loadedDiplomacy.getPendingTreatyOffer(0, 1),
                    "Pending treaty metadata should survive save/load.");
            assertEquals(Types.RELATIONSHIP.TREATY, loadedDiplomacy.getPendingOfferType(0, 1),
                    "Pending offer type should survive save/load.");
            assertTrue(loader.getTribes()[0].getKnownCapitalTribes().contains(1),
                    "Known capital metadata should survive save/load.");
        } finally {
            deleteRecursivelyQuietly(saveRoot);
        }
    }

    private static void testPeaceOffersCanEndWars() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        GameState gs = scenario.buildState();
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        diplomacy.setRelationship(0, 1, Types.RELATIONSHIP.WAR, gs.getTick());
        gs.computePlayerActions(gs.getActiveTribe());
        assertFalse(hasTribeActionType(gs, Types.ACTION.PROPOSE_PEACE),
                "Peace offers should stay locked before Strategy is researched.");

        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        gs = scenario.buildState();
        diplomacy = gs.getBoard().getDiplomacy();
        diplomacy.setRelationship(0, 1, Types.RELATIONSHIP.WAR, gs.getTick());
        gs.computePlayerActions(gs.getActiveTribe());
        assertTrue(hasTribeActionType(gs, Types.ACTION.PROPOSE_PEACE),
                "Strategy should unlock peace-offer actions while at war.");

        ProposePeace proposePeace = new ProposePeace(0);
        proposePeace.setTargetID(1);
        assertTrue(proposePeace.isFeasible(gs), "Propose peace should be feasible while at war.");
        assertTrue(new ProposePeaceCommand().execute(proposePeace, gs), "Propose peace should execute.");
        assertTrue(diplomacy.hasPendingOffer(0, 1, Types.RELATIONSHIP.PEACE),
                "Peace offer should become pending.");

        AcceptPeace acceptPeace = new AcceptPeace(1);
        acceptPeace.setTargetID(0);
        assertTrue(acceptPeace.isFeasible(gs), "Accept peace should be feasible for the target tribe.");
        assertTrue(new AcceptPeaceCommand().execute(acceptPeace, gs), "Accept peace should execute.");
        assertEquals(Types.RELATIONSHIP.PEACE, diplomacy.getRelationship(0, 1),
                "Accepting peace should end the war.");
        assertFalse(diplomacy.hasPendingOfferBetween(0, 1),
                "Accepting peace should clear pending offers.");
    }

    private static void testTreatyOffersCanBeAcceptedAndCanceled() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        GameState gs = scenario.buildState();
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();

        gs.computePlayerActions(gs.getActiveTribe());
        assertTrue(hasTribeActionType(gs, Types.ACTION.PROPOSE_TREATY),
                "Diplomacy tech should unlock treaty proposals.");

        ProposeTreaty proposeTreaty = new ProposeTreaty(0);
        proposeTreaty.setTargetID(1);
        assertTrue(proposeTreaty.isFeasible(gs), "Treaty proposal should be feasible from peace.");
        assertTrue(new ProposeTreatyCommand().execute(proposeTreaty, gs), "Treaty proposal should execute.");
        assertTrue(diplomacy.hasPendingOffer(0, 1, Types.RELATIONSHIP.TREATY),
                "Treaty offer should become pending.");

        AcceptTreaty acceptTreaty = new AcceptTreaty(1);
        acceptTreaty.setTargetID(0);
        assertTrue(acceptTreaty.isFeasible(gs), "Treaty target should be able to accept.");
        assertTrue(new AcceptTreatyCommand().execute(acceptTreaty, gs), "Accept treaty should execute.");
        assertEquals(Types.RELATIONSHIP.TREATY, diplomacy.getRelationship(0, 1),
                "Accepting a treaty should move the pair into treaty state.");

        CancelTreaty cancelTreaty = new CancelTreaty(0);
        cancelTreaty.setTargetID(1);
        assertTrue(cancelTreaty.isFeasible(gs), "Cancel treaty should be feasible while treaty is active.");
        assertTrue(new CancelTreatyCommand().execute(cancelTreaty, gs), "Cancel treaty should execute.");
        assertEquals(Types.RELATIONSHIP.PEACE, diplomacy.getRelationship(0, 1),
                "Cancel treaty should return the pair to peace.");
    }

    private static void testCancelTreatyKeepsEmbassiesAndDisbandsIntruders() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        Unit borderIntruder = scenario.addUnit(homeCapital, Types.UNIT.WARRIOR, 3, 4);
        Unit homeGuard = scenario.addUnit(homeCapital, Types.UNIT.WARRIOR, 1, 1);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[1].getTribesMet().add(0);
        scenario.tribes[0].setStars(10);
        scenario.tribes[1].setStars(10);
        GameState gs = scenario.buildState();

        BuildEmbassy embassy01 = new BuildEmbassy(0);
        embassy01.setTargetID(1);
        BuildEmbassy embassy10 = new BuildEmbassy(1);
        embassy10.setTargetID(0);
        assertTrue(new BuildEmbassyCommand().execute(embassy01, gs), "Embassy 0->1 should build before canceling a treaty.");
        assertTrue(new BuildEmbassyCommand().execute(embassy10, gs), "Embassy 1->0 should build before canceling a treaty.");

        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 3);
        borderIntruder.setStatus(Types.TURN_STATUS.FRESH);
        homeGuard.setStatus(Types.TURN_STATUS.FRESH);

        CancelTreaty cancelTreaty = new CancelTreaty(0);
        cancelTreaty.setTargetID(1);
        assertTrue(new CancelTreatyCommand().execute(cancelTreaty, gs), "Cancel treaty should execute while treaty is active.");

        assertEquals(Types.RELATIONSHIP.PEACE, gs.getBoard().getDiplomacy().getRelationship(0, 1),
                "Cancel treaty should move the relationship back to peace.");
        assertTrue(gs.getBoard().hasEmbassy(0, 1), "Breaking peace should not destroy the attacker's embassy.");
        assertTrue(gs.getBoard().hasEmbassy(1, 0), "Breaking peace should not destroy the defender's embassy.");
        assertTrue(gs.getActor(borderIntruder.getActorId()) == null,
                "Breaking peace should disband units that remain in the former ally's territory.");
        assertEquals(Types.TURN_STATUS.FINISHED, homeGuard.getStatus(),
                "Breaking peace should disable the remaining units for the rest of the turn.");
    }

    private static void testCancelTreatyFinishesRemainingUnits() {
        Scenario scenario = new Scenario(6, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 5, 5);
        Unit homeGuard = scenario.addUnit(homeCapital, Types.UNIT.WARRIOR, 1, 1);
        Unit neutralRider = scenario.addUnit(homeCapital, Types.UNIT.RIDER, 2, 2);
        GameState gs = scenario.buildState();

        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 3);
        homeGuard.setStatus(Types.TURN_STATUS.FRESH);
        neutralRider.setStatus(Types.TURN_STATUS.FRESH);

        CancelTreaty cancelTreaty = new CancelTreaty(0);
        cancelTreaty.setTargetID(1);
        assertTrue(new CancelTreatyCommand().execute(cancelTreaty, gs),
                "Cancel treaty should execute while treaty is active.");

        assertEquals(Types.TURN_STATUS.FINISHED, homeGuard.getStatus(),
                "Breaking peace should finish units in owned territory.");
        assertEquals(Types.TURN_STATUS.FINISHED, neutralRider.getStatus(),
                "Breaking peace should also finish units outside enemy territory.");
        assertTrue(gs.getActor(neutralRider.getActorId()) != null,
                "Units outside the former ally's borders should remain on the board.");
    }

    private static void testTreatyBlocksHostileLegalActions() {
        Scenario attackScenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackCity = attackScenario.addCapital(0, 0, 0);
        City defenseCity = attackScenario.addCapital(1, 4, 4);
        Unit attacker = attackScenario.addUnit(attackCity, Types.UNIT.WARRIOR, 2, 2);
        Unit defender = attackScenario.addUnit(defenseCity, Types.UNIT.WARRIOR, 2, 3);
        GameState attackState = attackScenario.buildState();
        attackState.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 3);
        attacker.setStatus(Types.TURN_STATUS.FRESH);
        defender.setStatus(Types.TURN_STATUS.FRESH);

        Attack attack = new Attack(attacker.getActorId());
        attack.setTargetId(defender.getActorId());
        assertFalse(attack.isFeasible(attackState), "Treaty should block direct attack feasibility.");
        attackState.computePlayerActions(attackState.getActiveTribe());
        assertFalse(hasUnitActionType(attackState, Types.ACTION.ATTACK), "Treaty should remove attack actions from legal generation.");

        Scenario captureScenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = captureScenario.addCapital(0, 0, 0);
        City defenderCapital = captureScenario.addCapital(1, 2, 2);
        Unit capturingUnit = captureScenario.addUnit(attackerCapital, Types.UNIT.WARRIOR, 2, 2);
        GameState captureState = captureScenario.buildState();
        captureState.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 3);
        capturingUnit.setStatus(Types.TURN_STATUS.FRESH);

        Capture capture = new Capture(capturingUnit.getActorId());
        capture.setCaptureType(Types.TERRAIN.CITY);
        capture.setTargetCity(defenderCapital.getActorId());
        assertFalse(capture.isFeasible(captureState), "Treaty should block city capture.");

        Scenario convertScenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City converterCity = convertScenario.addCapital(0, 0, 0);
        City targetCity = convertScenario.addCapital(1, 4, 4);
        Unit mindBender = convertScenario.addUnit(converterCity, Types.UNIT.MIND_BENDER, 2, 2);
        Unit target = convertScenario.addUnit(targetCity, Types.UNIT.WARRIOR, 2, 3);
        GameState convertState = convertScenario.buildState();
        convertState.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 3);
        mindBender.setStatus(Types.TURN_STATUS.FRESH);
        target.setStatus(Types.TURN_STATUS.FRESH);

        Convert convert = new Convert(mindBender.getActorId());
        convert.setTargetId(target.getActorId());
        assertFalse(convert.isFeasible(convertState), "Treaty should block conversion.");

        assertFalse(hasTribeActionType(convertState, Types.ACTION.PROPOSE_PEACE),
                "Treaty relationships should not expose peace-offer actions while the treaty is active.");
    }

    private static void testTreatyAlliesCanUseAlliedRoads() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        City alliedCapital = scenario.addCapital(1, 2, 2);
        Unit traveler = scenario.addUnit(homeCapital, Types.UNIT.WARRIOR, 1, 2);
        scenario.board.addRoad(1, 2);
        scenario.board.addRoad(3, 2);

        GameState gs = scenario.buildState();
        Unit alliedGuard = gs.getBoard().getUnitAt(alliedCapital.getPosition().x, alliedCapital.getPosition().y);
        if (alliedGuard != null) {
            gs.killUnit(alliedGuard);
        }
        traveler.setStatus(Types.TURN_STATUS.FRESH);

        MoveFactory moveFactory = new MoveFactory();
        assertFalse(hasMoveTo(moveFactory.computeActionVariants(traveler, gs), 3, 2),
                "Without a treaty, allied-border roads should not grant the movement bonus.");

        gs.getTribe(0).revealArea(gs.getBoard(), 2, 2, 1);
        gs.getBoard().getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY, 2);
        assertTrue(hasMoveTo(moveFactory.computeActionVariants(traveler, gs), 3, 2),
                "Treaty allies should be able to chain movement across allied roads.");
    }

    private static void testGeneratedLegalActionsStayFeasibleInDiplomacyScenario() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 2, 2);
        Unit cloak = scenario.addUnit(homeCapital, Types.UNIT.CLOAK, 2, 1);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        scenario.tribes[0].setStars(20);
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[1].getTribesMet().add(0);
        cloak.setStatus(Types.TURN_STATUS.FRESH);

        GameState gs = scenario.buildState();
        gs.computePlayerActions(gs.getActiveTribe());

        assertTrue(hasTribeActionType(gs, Types.ACTION.BUILD_EMBASSY),
                "Representative diplomacy scenario should expose embassy actions.");
        assertTrue(hasTribeActionType(gs, Types.ACTION.PROPOSE_TREATY),
                "Representative diplomacy scenario should expose treaty proposal actions.");
        assertTrue(hasUnitActionType(gs, Types.ACTION.INFILTRATE),
                "Representative diplomacy scenario should expose cloak infiltration.");
        assertTrue(hasSpawnAction(gs, homeCapital.getActorId(), Types.UNIT.CLOAK),
                "Representative diplomacy scenario should expose diplomacy-based cloak spawning.");

        for (Action action : gs.getAllAvailableActions()) {
            assertTrue(action.isFeasible(gs), "Generated action should remain feasible: " + action);
        }
    }

    private static void testDiplomacyUnlocksCloakSpawning() {
        Scenario lockedScenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City lockedCapital = lockedScenario.addCapital(0, 2, 2);
        lockedScenario.tribes[0].setStars(20);
        GameState lockedState = lockedScenario.buildState();
        lockedState.computePlayerActions(lockedState.getActiveTribe());

        assertFalse(hasSpawnAction(lockedState, lockedCapital.getActorId(), Types.UNIT.CLOAK),
                "Cloaks should stay unavailable before Diplomacy is researched.");

        Scenario unlockedScenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City unlockedCapital = unlockedScenario.addCapital(0, 2, 2);
        unlockedScenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        unlockedScenario.tribes[0].setStars(20);
        GameState unlockedState = unlockedScenario.buildState();
        unlockedState.computePlayerActions(unlockedState.getActiveTribe());

        assertTrue(hasSpawnAction(unlockedState, unlockedCapital.getActorId(), Types.UNIT.CLOAK),
                "Diplomacy should unlock cloak recruitment.");
    }

    private static void testHiddenCloakRedactedFromObservationCopy() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City cloakCity = scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        Unit cloak = scenario.addUnit(cloakCity, Types.UNIT.CLOAK, 1, 1);
        GameState gs = scenario.buildState();

        cloak.setStatus(Types.TURN_STATUS.FRESH);
        Move move = new Move(cloak.getActorId());
        move.setDestination(new Vector2d(2, 1));
        assertTrue(move.isFeasible(gs), "Cloak should be able to move onto an open land tile.");
        assertTrue(new MoveCommand().execute(move, gs), "Cloak move should execute.");
        assertTrue(cloak.isHidden(), "Cloak should become hidden after ending on open land.");

        GameState observerCopy = gs.copy(1);
        assertEquals(null, observerCopy.getBoard().getUnitAt(2, 1),
                "Observation copy should redact a hidden enemy cloak.");
        assertTrue(gs.getBoard().getUnitAt(2, 1) != null,
                "Authoritative state should still retain the hidden cloak.");
    }

    private static void testAdjacentUnitsDoNotRevealHiddenCloaks() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City cloakCity = scenario.addCapital(0, 0, 0);
        City observerCity = scenario.addCapital(1, 4, 4);
        Unit cloak = scenario.addUnit(cloakCity, Types.UNIT.CLOAK, 2, 2);
        scenario.addUnit(observerCity, Types.UNIT.WARRIOR, 3, 3);
        GameState gs = scenario.buildState();

        cloak.setHidden(true);
        GameState observerCopy = gs.copy(1);
        Unit visible = observerCopy.getBoard().getUnitAt(2, 2);
        assertEquals(null, visible, "Adjacent enemy units should only hint at a hidden cloak, not reveal it.");
        assertTrue(gs.getBoard().hasHiddenEnemyCloak(1, 2, 2),
                "The authoritative state should still treat the adjacent cloak as hidden.");
    }

    private static void testFreshlySpawnedCloaksStayVisibleUntilMove() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City cloakCity = scenario.addCapital(0, 2, 2);
        City observerCity = scenario.addCapital(1, 4, 4);
        Unit cloak = scenario.addUnit(cloakCity, Types.UNIT.CLOAK, 2, 2);
        scenario.addUnit(observerCity, Types.UNIT.WARRIOR, 3, 2);
        GameState gs = scenario.buildState();

        assertFalse(cloak.isHidden(), "Freshly spawned cloaks should start visible.");
        GameState observerCopy = gs.copy(1);
        Unit visible = observerCopy.getBoard().getUnitAt(2, 2);
        assertTrue(visible != null, "Freshly spawned cloaks should remain visible to enemies.");
        assertEquals(Types.UNIT.CLOAK, visible.getType(), "Visible spawned cloak should still identify as a cloak.");
    }

    private static void testExploredTilesPersistWithoutVisibilityRefresh() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 4);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.FOREST);
        scenario.addUnit(enemyCapital, Types.UNIT.WARRIOR, 3, 3);
        GameState gs = scenario.buildState();

        Tribe observer = gs.getTribe(0);
        observer.revealArea(gs.getBoard(), 3, 3, 0);
        assertTrue(observer.isExplored(3, 3), "Manual reveal should mark the tile explored.");
        assertTrue(observer.isExplored(3, 3), "Explored state should persist without any visibility refresh cycle.");

        GameState observerCopy = gs.copy(0);
        assertEquals(Types.TERRAIN.FOREST, observerCopy.getBoard().getTerrainAt(3, 3),
                "Observation copies should preserve explored terrain even after sight is lost.");
        assertEquals(Types.UNIT.WARRIOR, observerCopy.getBoard().getUnitAt(3, 3).getType(),
                "Observation copies should preserve normal units on explored tiles even after sight is lost.");
    }

    private static void testVisibilityMapUsesExplorationState() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        scenario.addCapital(0, 0, 0);
        scenario.board.setTerrainAt(4, 4, Types.TERRAIN.MOUNTAIN);
        GameState gs = scenario.buildState();

        Tribe tribe = gs.getTribe(0);
        tribe.revealArea(gs.getBoard(), 4, 4, 0);
        assertTrue(tribe.isExplored(4, 4), "Reveal should mark the remote tile explored.");
        assertTrue(gs.getVisibilityMap()[4][4], "Visibility map should reflect explored tiles.");
    }

    private static void testScoutAndCloakRevealRange() {
        Scenario scoutScenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City scoutCapital = scoutScenario.addCapital(0, 0, 0);
        scoutScenario.addUnit(scoutCapital, Types.UNIT.SCOUT, 3, 3);
        GameState scoutState = scoutScenario.buildState();
        Tribe scoutTribe = scoutState.getTribe(0);
        scoutTribe.clearView(3, 3, 2, scoutState.getRandomGenerator(), scoutState.getBoard());
        assertTrue(scoutTribe.isExplored(5, 5), "Scout units should reveal tiles up to two tiles away.");
        assertFalse(scoutTribe.isExplored(6, 6), "Scout reveal should not extend beyond a two-tile radius.");

        Scenario cloakScenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City cloakCapital = cloakScenario.addCapital(0, 0, 0);
        cloakScenario.addUnit(cloakCapital, Types.UNIT.CLOAK, 3, 3);
        GameState cloakState = cloakScenario.buildState();
        Tribe cloakTribe = cloakState.getTribe(0);
        cloakTribe.clearView(3, 3, 2, cloakState.getRandomGenerator(), cloakState.getBoard());
        assertTrue(cloakTribe.isExplored(5, 5), "Cloaks should use the same extended scout reveal.");
        assertFalse(cloakTribe.isExplored(6, 6), "Cloak reveal should still stop at a two-tile radius.");
    }

    private static void testMountainRevealRange() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.MOUNTAIN);
        scenario.addUnit(capital, Types.UNIT.WARRIOR, 3, 3);
        GameState gs = scenario.buildState();
        Tribe tribe = gs.getTribe(0);
        tribe.clearView(3, 3, 2, gs.getRandomGenerator(), gs.getBoard());
        assertTrue(tribe.isExplored(5, 5), "Mountains should extend normal reveal by one tile.");
        assertFalse(tribe.isExplored(6, 6), "Mountain reveal should stop at a two-tile radius.");
    }

    private static void testMoveGenerationRequiresExploredDestination() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        Unit rider = scenario.addUnit(capital, Types.UNIT.RIDER, 1, 1);
        GameState gs = scenario.buildState();

        rider.setStatus(Types.TURN_STATUS.FRESH);
        Tribe tribe = gs.getTribe(0);
        tribe.revealArea(gs.getBoard(), 3, 1, 0);
        assertTrue(tribe.isExplored(3, 1), "Manual reveal should mark the destination explored.");

        gs.computePlayerActions(tribe);
        Move move = new Move(rider.getActorId());
        move.setDestination(new Vector2d(3, 1));
        assertTrue(hasMoveTo(gs.getUnitActions(rider.getActorId()), 3, 1),
                "Explored destination tiles should still be offered as move destinations when not under clouded fog.");
        assertTrue(move.isFeasible(gs),
                "Move feasibility should accept explored targets even when they are not currently in line of sight.");
    }

    private static void testObservationCopiesDoNotInventEnemyRoadAccess() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City homeCapital = scenario.addCapital(0, 0, 0);
        City enemyCapital = scenario.addCapital(1, 4, 2);
        scenario.board.addRoad(3, 2);
        scenario.board.getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.WAR);

        GameState gs = scenario.buildState();
        gs.getTribe(0).revealArea(gs.getBoard(), 3, 2, 0);

        Board authoritativeBoard = gs.getBoard();
        int enemyCityId = authoritativeBoard.getCityIdAt(3, 2);
        assertEquals(enemyCapital.getActorId(), enemyCityId,
                "The revealed border road should still belong to the hidden enemy city.");
        assertTrue(authoritativeBoard.getActor(enemyCityId) instanceof City,
                "The authoritative board should still have the owning city actor.");
        assertFalse(authoritativeBoard.canUseRoad(0, 3, 2),
                "Enemy roads without a treaty should not be usable in the authoritative state.");

        GameState observerCopy = gs.copy(0);
        Board observerBoard = observerCopy.getBoard();
        assertEquals(enemyCityId, observerBoard.getCityIdAt(3, 2),
                "Observation copies should preserve tile ownership metadata for explored tiles.");
        assertEquals(null, observerBoard.getActor(enemyCityId),
                "The owning enemy city actor should stay hidden when its center is unexplored.");
        assertFalse(observerBoard.canUseRoad(0, 3, 2),
                "Observation copies should not fabricate road access when the owning city actor is hidden.");
    }

    private static void testMovingIntoHiddenCloakRevealsAndBlocksMovement() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City cloakCity = scenario.addCapital(0, 0, 0);
        City moverCity = scenario.addCapital(1, 4, 4);
        Unit cloak = scenario.addUnit(cloakCity, Types.UNIT.CLOAK, 2, 2);
        Unit mover = scenario.addUnit(moverCity, Types.UNIT.RIDER, 0, 2);
        GameState gs = scenario.buildState();

        cloak.setHidden(true);
        mover.setStatus(Types.TURN_STATUS.FRESH);
        Move move = new Move(mover.getActorId());
        move.setDestination(new Vector2d(2, 2));
        gs.getTribe(1).revealArea(gs.getBoard(), 2, 2, 0);
        assertTrue(move.isFeasible(gs), "Hidden cloaks should not block move feasibility before reveal.");
        assertTrue(new MoveCommand().execute(move, gs),
                "Moving into a hidden cloak should count as a handled action because it reveals the hidden unit.");
        assertVectorEquals(new Vector2d(0, 2), mover.getPosition(), "Blocked mover should stay in place.");
        assertFalse(cloak.isHidden(), "Collision reveal should unhide the cloak.");
        assertEquals(Types.TURN_STATUS.FRESH, mover.getStatus(),
                "Reveal-only collisions should not consume the mover's turn.");
    }

    private static void testMoveCommandRejectsOccupiedDestination() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City moverCity = scenario.addCapital(0, 0, 0);
        City blockerCity = scenario.addCapital(1, 4, 4);
        Unit mover = scenario.addUnit(moverCity, Types.UNIT.RIDER, 0, 2);
        Unit blocker = scenario.addUnit(blockerCity, Types.UNIT.WARRIOR, 2, 2);
        GameState gs = scenario.buildState();

        mover.setStatus(Types.TURN_STATUS.FRESH);
        gs.getTribe(0).revealArea(gs.getBoard(), 2, 2, 0);

        Move forcedMove = new Move(mover.getActorId()) {
            @Override
            public boolean isFeasible(final GameState ignored) {
                return true;
            }
        };
        forcedMove.setDestination(new Vector2d(2, 2));

        assertFalse(new MoveCommand().execute(forcedMove, gs),
                "Authoritative execution should reject moves into occupied destinations.");
        assertVectorEquals(new Vector2d(0, 2), mover.getPosition(), "Rejected moves should leave the mover in place.");
        assertEquals(blocker.getActorId(), gs.getBoard().getUnitAt(2, 2).getActorId(),
                "Rejected moves should leave the blocking unit in place.");
    }

    private static void testScoutDisembarkKeepsScoutRevealRange() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 3);
        scenario.board.setTerrainAt(1, 3, Types.TERRAIN.DEEP_WATER);
        Unit scout = scenario.addUnit(capital, Types.UNIT.SCOUT, 1, 3);
        ((CarrierUnit) scout).setBaseLandUnit(Types.UNIT.WARRIOR);
        GameState gs = scenario.buildState();

        Tribe tribe = scenario.tribes[0];
        scout.setStatus(Types.TURN_STATUS.FRESH);
        assertFalse(tribe.isExplored(4, 3),
                "A tile three columns away should start outside the scout's initial reveal radius.");

        Move move = new Move(scout.getActorId());
        move.setDestination(new Vector2d(2, 3));
        assertTrue(move.isFeasible(gs), "Scout should be able to disembark onto adjacent land.");
        assertTrue(new MoveCommand().execute(move, gs), "Scout disembark move should execute.");

        Unit landed = gs.getBoard().getUnitAt(2, 3);
        assertEquals(Types.UNIT.WARRIOR, landed.getType(),
                "Disembarking a scout should convert it back into its carried land unit.");
        assertTrue(tribe.isExplored(4, 3),
                "Scout disembark should reveal using the scout's radius before reverting to the land unit.");
        assertFalse(tribe.isExplored(5, 3),
                "Scout disembark reveal should still stop outside the two-tile scout radius.");
    }

    private static void testInfiltrationSpawnsDaggersAndStartsWar() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = scenario.addCapital(0, 0, 0);
        City defenderCapital = scenario.addCapital(1, 2, 2);
        Unit cloak = scenario.addUnit(attackerCapital, Types.UNIT.CLOAK, 2, 1);
        defenderCapital.levelUp();
        defenderCapital.levelUp();
        scenario.tribes[0].getTribesMet().add(1);
        scenario.tribes[1].getTribesMet().add(0);
        cloak.setStatus(Types.TURN_STATUS.FRESH);
        GameState gs = scenario.buildState();
        int starsBefore = scenario.tribes[0].getStars();

        gs.computePlayerActions(scenario.tribes[0]);
        assertTrue(hasUnitActionType(gs, Types.ACTION.INFILTRATE),
                "Adjacent cloaks should generate infiltrate actions.");

        Infiltrate infiltrate = new Infiltrate(cloak.getActorId());
        infiltrate.setTargetCityId(defenderCapital.getActorId());
        assertTrue(infiltrate.isFeasible(gs), "Infiltration should be feasible from an adjacent cloak.");
        assertTrue(new InfiltrateCommand().execute(infiltrate, gs), "Infiltration should execute.");

        assertEquals(Types.RELATIONSHIP.WAR, gs.getBoard().getDiplomacy().getRelationship(0, 1),
                "Infiltration should force the tribes into war.");
        assertEquals(starsBefore + defenderCapital.getProduction(), scenario.tribes[0].getStars(),
                "Infiltration should steal stars equal to the target city's production.");
        assertEquals(null, gs.getActor(cloak.getActorId()), "Cloak should be consumed by infiltration.");
        assertEquals(3, countExtraUnitsOfType(gs, 0, Types.UNIT.DAGGER),
                "City level should determine how many daggers are spawned.");
    }

    private static void testUnitStatsMatchWiki() {
        Unit knight = Types.UNIT.createUnit(new Vector2d(0, 0), 0, false, -1, 0, Types.UNIT.KNIGHT);
        Unit cloak = Types.UNIT.createUnit(new Vector2d(0, 0), 0, false, -1, 0, Types.UNIT.CLOAK);
        Unit dinghy = Types.UNIT.createUnit(new Vector2d(0, 0), 0, false, -1, 0, Types.UNIT.DINGHY);

        assertEquals(3.5, knight.getAttackValue(), 0.0001, "Knight attack should match the current wiki value.");
        assertEquals(10, knight.getMaxHP(), "Knight HP should match the current wiki value.");
        assertEquals(0.5, cloak.getDefenceValue(), 0.0001, "Cloak defence should match the current wiki value.");
        assertEquals(0.5, dinghy.getDefenceValue(), 0.0001, "Dinghy defence should match the current wiki value.");
    }

    private static void testCloakCannotBecomeVeteran() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        Unit cloak = scenario.addUnit(capital, Types.UNIT.CLOAK, 2, 3);
        GameState gs = scenario.buildState();

        cloak.setKills(TribesConfig.VETERAN_KILLS);
        MakeVeteran action = new MakeVeteran(cloak.getActorId());
        assertFalse(action.isFeasible(gs), "Cloaks should not be promotable to veterans.");
        assertFalse(new MakeVeteranCommand().execute(action, gs), "Veteran promotion should reject cloaks.");
    }

    private static void testDisbandRefundsMatchWiki() {
        Scenario scenario = new Scenario(6, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FREE_SPIRIT);

        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 3);
        warrior.setStatus(Types.TURN_STATUS.FRESH);
        int warriorScoreBefore = scenario.tribes[0].getScore();
        int starsBefore = scenario.tribes[0].getStars();
        Disband warriorDisband = new Disband(warrior.getActorId());
        assertTrue(warriorDisband.isFeasible(gs), "Fresh units with Free Spirit should be able to disband.");
        assertTrue(new DisbandCommand().execute(warriorDisband, gs), "Warrior disband should execute.");
        assertEquals(starsBefore + 1, scenario.tribes[0].getStars(),
                "Disbanding a warrior should return half its cost rounded down.");
        assertEquals(warriorScoreBefore - Types.UNIT.WARRIOR.getPoints(), scenario.tribes[0].getScore(),
                "Disbanding should remove the unit's score value.");

        Unit bomber = scenario.addUnit(capital, Types.UNIT.BOMBER, 3, 3);
        ((CarrierUnit) bomber).setBaseLandUnit(Types.UNIT.WARRIOR);
        bomber.setStatus(Types.TURN_STATUS.FRESH);
        Disband bomberDisband = new Disband(bomber.getActorId());
        assertTrue(new DisbandCommand().execute(bomberDisband, gs), "Bomber disband should execute.");
        assertEquals(starsBefore + 2, scenario.tribes[0].getStars(),
                "Disbanding a Bomber should refund only half the carried land unit cost, not the naval upgrade.");

        Unit juggernaut = scenario.addUnit(capital, Types.UNIT.JUGGERNAUT, 4, 3);
        ((CarrierUnit) juggernaut).setBaseLandUnit(Types.UNIT.SUPERUNIT);
        juggernaut.setStatus(Types.TURN_STATUS.FRESH);
        Disband juggernautDisband = new Disband(juggernaut.getActorId());
        assertTrue(new DisbandCommand().execute(juggernautDisband, gs), "Juggernaut disband should execute.");
        assertEquals(starsBefore + 7, scenario.tribes[0].getStars(),
                "Disbanding a Juggernaut should return five stars from the carried super unit only.");
    }

    private static void testDaggerAttacksSkipRetaliation() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City daggerCity = scenario.addCapital(0, 0, 0);
        City defenderCity = scenario.addCapital(1, 4, 4);
        Unit dagger = scenario.addUnit(daggerCity, Types.UNIT.DAGGER, 2, 2);
        Unit defender = scenario.addUnit(defenderCity, Types.UNIT.WARRIOR, 2, 3);
        GameState gs = scenario.buildState();

        dagger.setStatus(Types.TURN_STATUS.FRESH);
        defender.setStatus(Types.TURN_STATUS.FRESH);
        Attack attack = new Attack(dagger.getActorId());
        attack.setTargetId(defender.getActorId());
        AttackCommand command = new AttackCommand();

        assertTrue(attack.isFeasible(gs), "Dagger attack should be feasible at melee range.");
        assertFalse(command.isRetaliation(attack, gs), "Daggers should not trigger retaliation previews.");
        assertTrue(command.execute(attack, gs), "Dagger attack should execute.");
        assertEquals(dagger.getMaxHP(), dagger.getCurrentHP(), "Daggers should not take retaliation damage.");
        assertEquals(5, defender.getCurrentHP(), "Defender should still take the dagger's attack damage.");
    }

    private static void testRetaliationIgnoresAttackerVisibility() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = scenario.addCapital(0, 6, 6);
        City defenderCapital = scenario.addCapital(1, 0, 0);
        scenario.board.setTerrainAt(4, 4, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(2, 2, Types.TERRAIN.DEEP_WATER);
        Unit attacker = scenario.addUnit(attackerCapital, Types.UNIT.SCOUT, 4, 4);
        Unit defender = scenario.addUnit(defenderCapital, Types.UNIT.ARCHER, 2, 2);
        GameState gs = scenario.buildState();

        attacker.setStatus(Types.TURN_STATUS.FRESH);
        defender.setStatus(Types.TURN_STATUS.FRESH);
        Attack attack = new Attack(attacker.getActorId());
        attack.setTargetId(defender.getActorId());
        AttackCommand command = new AttackCommand();

        assertTrue(attack.isFeasible(gs), "Attack should still be feasible from outside the defender's vision.");
        assertTrue(command.isRetaliation(attack, gs), "Units should retaliate when the attack is legal and in range.");
        int attackerHpBefore = attacker.getCurrentHP();
        assertTrue(command.execute(attack, gs), "Fog attack should execute.");
        assertTrue(attacker.getCurrentHP() < attackerHpBefore, "Legal in-range attacks should still take retaliation damage.");
    }

    private static void testRuinStarsAlwaysGrantTenStars() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        scenario.board.setResourceAt(2, 2, Types.RESOURCE.RUINS);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 2);
        warrior.setStatus(Types.TURN_STATUS.FRESH);
        GameState gs = scenario.buildState(new FixedRandom(0));
        int starsBefore = scenario.tribes[0].getStars();

        Examine examine = new Examine(warrior.getActorId());
        assertTrue(new ExamineCommand().execute(examine, gs), "Land ruin examine should execute for the stars reward.");
        assertEquals(starsBefore + Types.EXAMINE_BONUS.RESOURCES.getBonus(), scenario.tribes[0].getStars(),
                "Ruin star rewards should always grant ten stars.");
        assertEquals(Types.TURN_STATUS.FINISHED, warrior.getStatus(),
                "Examining a ruin for stars should finish the unit's turn.");
    }

    private static void testRuinPopulationGivesThreeCapitalPopulation() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        scenario.board.setResourceAt(2, 2, Types.RESOURCE.RUINS);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 2);
        warrior.setStatus(Types.TURN_STATUS.FRESH);
        GameState gs = scenario.buildState(new FixedRandom(2));
        int populationBefore = capital.getPopulation();

        Examine examine = new Examine(warrior.getActorId());
        assertTrue(new ExamineCommand().execute(examine, gs), "Land ruin examine should execute for the population reward.");
        assertEquals(populationBefore + Types.EXAMINE_BONUS.POP_GROWTH.getBonus(), capital.getPopulation(),
                "Ruin population rewards should add three population to the capital.");
    }

    private static void testRuinResearchGrantsExactlyOneTechnology() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        scenario.board.setResourceAt(2, 2, Types.RESOURCE.RUINS);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 2);
        warrior.setStatus(Types.TURN_STATUS.FRESH);
        int researchedBefore = countResearchedTechs(scenario.tribes[0].getTechTree());

        GameState gs = scenario.buildState(new FixedRandom(1, 0));
        Examine examine = new Examine(warrior.getActorId());
        assertTrue(new ExamineCommand().execute(examine, gs), "Land ruin examine should execute for the research reward.");
        assertEquals(researchedBefore + 1, countResearchedTechs(scenario.tribes[0].getTechTree()),
                "A research ruin should grant exactly one additional technology.");
    }

    private static void testWaterRuinSpawnsVeteranRammer() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 0);
        scenario.board.setTerrainAt(2, 2, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(1, 2, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(3, 2, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(1, 3, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.DEEP_WATER);
        scenario.board.setResourceAt(2, 2, Types.RESOURCE.RUINS);
        Unit raft = scenario.addUnit(capital, Types.UNIT.RAFT, 2, 2);
        raft.setStatus(Types.TURN_STATUS.FRESH);

        Random riggedRandom = new Random() {
            @Override
            public int nextInt(int bound) {
                return bound - 1;
            }
        };
        scenario.board.setActiveTribeID(0);
        GameState gs = new GameState(riggedRandom, Types.GAME_MODE.SCORE, scenario.tribes, scenario.board, 0);
        Examine examine = new Examine(raft.getActorId());
        assertTrue(examine.isFeasible(gs), "A fresh unit on a ruin should be able to examine it.");
        assertTrue(new ExamineCommand().execute(examine, gs), "Water ruin examine should execute.");

        Unit reward = gs.getBoard().getUnitAt(2, 2);
        Unit pushedRaft = gs.getBoard().getUnitAt(2, 3);
        assertTrue(reward != null && reward.getType() == Types.UNIT.RAMMER,
                "Water ruins should be able to spawn a Rammer.");
        assertTrue(reward.isVeteran(), "Water ruin Rammer reward should be veteran.");
        assertEquals(15, reward.getCurrentHP(), "Veteran water ruin Rammer should start at 15 HP.");
        assertTrue(reward instanceof core.actors.units.CarrierUnit &&
                        ((core.actors.units.CarrierUnit) reward).getBaseLandUnit() == Types.UNIT.WARRIOR,
                "Water ruin Rammer should carry a Warrior.");
        assertTrue(pushedRaft != null && pushedRaft.getActorId() == raft.getActorId(),
                "Spawning the water ruin reward should push the examining raft aside.");
        assertTrue(gs.getBoard().getResourceAt(2, 2) == null, "Examining a ruin should clear the ruin resource.");
    }

    private static void testTempleScoringMatchesWiki() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        Tribe tribe = scenario.tribes[0];

        tribe.setStars(50);
        int scoreBefore = tribe.getScore();
        addBuilding(gs, capital, Types.BUILDING.TEMPLE, 2, 3);
        assertEquals(scoreBefore + 105, tribe.getScore(), "Building a temple should grant one hundred score plus one population.");

        Temple temple = (Temple) capital.getBuilding(2, 3);
        int turnScore = 0;
        for (int i = 0; i < 8; i++) {
            turnScore += temple.newTurn();
        }

        assertEquals(5, temple.getLevel(), "Temple should reach level five after four scoring steps.");
        assertEquals(400, turnScore, "Temple turn growth should add one hundred score per level-up.");
    }

    private static void testBuildingScoreUpdatesCityPointsWorth() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        Tribe tribe = scenario.tribes[0];

        int basePointsWorth = capital.getPointsWorth();
        addBuilding(gs, capital, Types.BUILDING.TEMPLE, 2, 3);
        assertEquals(basePointsWorth + 105, capital.getPointsWorth(),
                "Building a temple should add both score and population points to city points-worth.");

        Building temple = capital.getBuilding(2, 3);
        capital.removeBuilding(gs, temple);
        assertEquals(basePointsWorth, capital.getPointsWorth(),
                "Removing a temple should remove its score and population value from city points-worth.");

        int scoreBeforeMonument = tribe.getScore();
        addBuilding(gs, capital, Types.BUILDING.ALTAR_OF_PEACE, 3, 2);
        assertEquals(basePointsWorth + 415, capital.getPointsWorth(),
                "Building a monument should add monument score and population points to city points-worth.");
        assertEquals(scoreBeforeMonument + 415, tribe.getScore(),
                "Building a monument should add monument score plus population score.");

        Building monument = capital.getBuilding(3, 2);
        capital.removeBuilding(gs, monument);
        assertEquals(basePointsWorth, capital.getPointsWorth(),
                "Removing a monument should restore the previous city points-worth.");
    }

    private static void testHiddenCloakStateSurvivesSaveLoad() throws IOException {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 1);
        Unit cloak = scenario.addUnit(capital, Types.UNIT.CLOAK, 2, 1);
        GameState gs = scenario.buildState();

        cloak.setHidden(true);
        cloak.setCurrentHP(3);

        long seed = 9090L;
        Path saveRoot = Paths.get("save", Long.toString(seed));
        Path saveFile = saveRoot.resolve(gs.getTick() + "_" + gs.getActiveTribeID()).resolve("game.json");

        try {
            Files.createDirectories(saveRoot);
            GameSaver.writeTurnFile(gs, scenario.board, seed);

            GameLoader loader = new GameLoader(saveFile.toString());
            Unit loadedCloak = (Unit) loader.getBoard().getActor(cloak.getActorId());

            assertTrue(loadedCloak.isHidden(), "Hidden cloak flag should survive save/load.");
            assertEquals(cloak.getCurrentHP(), loadedCloak.getCurrentHP(), "Hidden cloak HP should survive save/load.");
            assertEquals(Types.UNIT.CLOAK, loadedCloak.getType(), "Loaded hidden unit should remain a cloak.");
        } finally {
            deleteRecursivelyQuietly(saveRoot);
        }
    }

    private static void testCityLevelUpBehavior() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();

        Tribe tribe = scenario.tribes[0];
        int productionBefore = capital.getProduction();
        int scoreBefore = tribe.getScore();

        capital.setPopulation(capital.getPopulation_need());

        LevelUp action = new LevelUp(capital.getActorId());
        action.setBonus(Types.CITY_LEVEL_UP.WORKSHOP);
        action.setTargetPos(capital.getPosition().copy());

        assertTrue(action.isFeasible(gs), "Workshop level-up should be feasible at city level 1");
        assertTrue(new LevelUpCommand().execute(action, gs), "Level-up command should execute");

        assertEquals(2, capital.getLevel(), "City should advance to level 2");
        assertEquals(0, capital.getPopulation(), "Population overflow should be consumed on level-up");
        assertEquals(3, capital.getPopulation_need(), "Next population threshold should update");
        assertEquals(productionBefore + 2, capital.getProduction(), "Workshop level-up should increase total production by level gain + workshop bonus");
        assertEquals(scoreBefore + Types.CITY_LEVEL_UP.WORKSHOP.getLevelUpPoints(1), tribe.getScore(), "Level-up points should be awarded to the tribe");
    }

    private static void testCityLevelUpPointsDeclineAfterLevelFive() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        Tribe tribe = scenario.tribes[0];

        tribe.setScore(0);
        capital.setPopulation(0);

        while (capital.getLevel() < 4) {
            capital.setPopulation(capital.getPopulation_need());
            LevelUp action = new LevelUp(capital.getActorId());
            action.setTargetPos(capital.getPosition().copy());
            if (capital.getLevel() == 1) {
                action.setBonus(Types.CITY_LEVEL_UP.WORKSHOP);
            } else if (capital.getLevel() == 2) {
                action.setBonus(Types.CITY_LEVEL_UP.RESOURCES);
            } else {
                action.setBonus(Types.CITY_LEVEL_UP.POP_GROWTH);
            }
            assertTrue(new LevelUpCommand().execute(action, gs), "Valid level-up should execute while climbing to level four.");
        }

        capital.setPopulation(capital.getPopulation_need());
        LevelUp levelFivePark = new LevelUp(capital.getActorId());
        levelFivePark.setBonus(Types.CITY_LEVEL_UP.PARK);
        levelFivePark.setTargetPos(capital.getPosition().copy());
        assertTrue(new LevelUpCommand().execute(levelFivePark, gs), "Level-five park upgrade should execute.");

        int scoreBefore = tribe.getScore();
        capital.setPopulation(capital.getPopulation_need());
        LevelUp levelSixPark = new LevelUp(capital.getActorId());
        levelSixPark.setBonus(Types.CITY_LEVEL_UP.PARK);
        levelSixPark.setTargetPos(capital.getPosition().copy());
        assertTrue(new LevelUpCommand().execute(levelSixPark, gs), "Level-six park upgrade should execute.");

        assertEquals(scoreBefore + 20 + TribesConfig.CITY_LEVEL_UP_PARK, tribe.getScore(),
                "City level-up score should keep declining after level five.");
    }

    private static void testBorderGrowthRevealsExpandedBorderLighthouses() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        scenario.board.setTerrainAt(0, 0, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setResourceAt(0, 0, Types.RESOURCE.LIGHTHOUSE);
        GameState gs = scenario.buildState();

        Tribe tribe = scenario.tribes[0];
        capital.setPopulation(capital.getPopulation_need());
        capital.levelUp();
        capital.setPopulation(capital.getPopulation_need());
        capital.levelUp();
        capital.setPopulation(capital.getPopulation_need());

        LevelUp action = new LevelUp(capital.getActorId());
        action.setBonus(Types.CITY_LEVEL_UP.BORDER_GROWTH);
        action.setTargetPos(capital.getPosition().copy());

        assertTrue(new LevelUpCommand().execute(action, gs), "Border growth level-up should execute.");
        assertTrue(tribe.isExplored(0, 0), "Border growth should reveal newly expanded border tiles.");
        assertEquals(1, tribe.getDiscoveredLighthouses().size(),
                "Border growth should discover lighthouses in the expanded border.");
        assertEquals(1, capital.getPopulation(),
                "Discovered lighthouse should add one population to the capital after level-up population is spent.");
    }

    private static void testVisibleLighthousesShowDiscoverers() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI);
        scenario.addCapital(0, 2, 2);
        scenario.addCapital(1, 4, 4);
        scenario.addCapital(2, 0, 4);
        scenario.board.setResourceAt(4, 0, Types.RESOURCE.LIGHTHOUSE);
        GameState gs = scenario.buildState();

        scenario.tribes[0].clearView(4, 0, 0, gs.getRandomGenerator(), gs.getBoard());
        scenario.tribes[2].clearView(4, 0, 0, gs.getRandomGenerator(), gs.getBoard());
        scenario.tribes[1].revealArea(gs.getBoard(), 4, 0, 0);

        GameState observerCopy = gs.copy(1);

        ArrayList<Integer> discoverers = observerCopy.getBoard().getLighthouseDiscoverers(4, 0);
        assertEquals(2, discoverers.size(), "Visible lighthouses should expose all tribes that have discovered them.");
        assertTrue(discoverers.contains(0), "Visible lighthouse should include Imperius as a discoverer.");
        assertTrue(discoverers.contains(2), "Visible lighthouse should include Xin-Xi as a discoverer.");
    }

    private static void testNegativePopulationReducesCityProduction() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();

        capital.addPopulation(scenario.tribes[0], -1);
        assertEquals(1, capital.getProduction(),
                "A capital with one negative population should lose exactly one star of production.");

        capital.addPopulation(scenario.tribes[0], -2);

        assertEquals(-3, capital.getPopulation(),
                "City population should be allowed to remain below negative city level.");
        assertEquals(0, capital.getProduction(),
                "City production should clamp at zero only after negative population cancels all income.");
        assertEquals(0, scenario.tribes[0].getMaxProduction(gs), "Tribe production should not become negative from a starving city.");
    }

    private static void testRecoverGetsTerritoryBonusInBorders() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 3);
        GameState gs = scenario.buildState();

        warrior.setCurrentHP(4);
        warrior.setStatus(Types.TURN_STATUS.FRESH);

        Recover recover = new Recover(warrior.getActorId());
        assertTrue(recover.isFeasible(gs), "Damaged fresh units should be able to recover.");
        assertTrue(new RecoverCommand().execute(recover, gs), "Recover should execute in owned territory.");
        assertEquals(8, warrior.getCurrentHP(), "Recover should gain the territory bonus anywhere inside owned borders.");
    }

    private static void testMarketIncomeMatchesSupportBuildingLevels() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 3, 3);
        GameState gs = scenario.buildState();

        addBuilding(gs, capital, Types.BUILDING.MINE, 2, 4);
        addBuilding(gs, capital, Types.BUILDING.FORGE, 3, 4);
        addBuilding(gs, capital, Types.BUILDING.MARKET, 4, 4);

        assertEquals(1, gs.getBoard().getMarketIncomeForTribe(0),
                "A market next to a level-1 forge should add one star per turn.");

        addBuilding(gs, capital, Types.BUILDING.LUMBER_HUT, 4, 2);
        addBuilding(gs, capital, Types.BUILDING.SAWMILL, 4, 3);

        assertEquals(2, gs.getBoard().getMarketIncomeForTribe(0),
                "Adding a level-1 sawmill next to the market should raise the total to two stars per turn.");
    }

    private static void testMarketBuildRequiresAdjacentFriendlySupport() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City capital = scenario.addCapital(0, 3, 3);
        City enemyCity = scenario.addCity(1, 2, 6);
        GameState gs = scenario.buildState();

        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.TRADE);
        scenario.tribes[0].setStars(20);

        Build market = new Build(capital.getActorId());
        market.setBuildingType(Types.BUILDING.MARKET);
        market.setTargetPos(new Vector2d(2, 4));
        assertFalse(market.isFeasible(gs),
                "Markets should require at least one adjacent friendly support building.");

        addBuilding(gs, enemyCity, Types.BUILDING.FORGE, 2, 5);
        assertFalse(market.isFeasible(gs),
                "Enemy support buildings should not satisfy market placement.");

        addBuilding(gs, capital, Types.BUILDING.MINE, 1, 4);
        addBuilding(gs, capital, Types.BUILDING.FORGE, 3, 4);
        assertTrue(market.isFeasible(gs),
                "Friendly adjacent support buildings should satisfy market placement.");
    }

    private static void testRoadsCanBeBuiltOnExploredFoggedTiles() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        scenario.addCapital(0, 0, 0);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(TribesConfig.ROAD_COST);
        GameState gs = scenario.buildState();

        Tribe tribe = gs.getTribe(0);
        tribe.revealArea(gs.getBoard(), 3, 3, 0);
        assertTrue(tribe.isExplored(3, 3), "Manual reveal should mark the road tile explored.");

        BuildRoad buildRoad = new BuildRoad(0);
        buildRoad.setPosition(new Vector2d(3, 3));
        assertTrue(buildRoad.isFeasible(gs), "Roads should still be buildable on explored tiles under fog.");
        assertTrue(new BuildRoadCommand().execute(buildRoad, gs), "Road build should execute on an explored fogged tile.");
        assertTrue(gs.getBoard().checkTradeNetwork(3, 3), "Built road should register in the trade network.");
    }

    private static void testBridgeBuildsOnExploredNeutralWaterForFiveStars() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        scenario.addCapital(0, 1, 3);
        scenario.addCity(0, 5, 3);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        GameState gs = scenario.buildState();

        scenario.tribes[0].clearView(3, 3, 1, gs.getRandomGenerator(), gs.getBoard());
        int starsBefore = scenario.tribes[0].getStars();

        BuildRoad buildBridge = new BuildRoad(0);
        buildBridge.setPosition(new Vector2d(3, 3));

        assertEquals(TribesConfig.BRIDGE_COST, gs.getBoard().getRoadCostAt(3, 3),
                "Water road tiles should charge the bridge cost.");
        assertTrue(buildBridge.isFeasible(gs),
                "Explored neutral water between opposite land tiles should allow bridge construction.");
        assertTrue(new BuildRoadCommand().execute(buildBridge, gs),
                "Bridge build should execute on valid neutral water.");
        assertTrue(gs.getBoard().isBridge(3, 3), "Built water roads should register as bridges.");
        assertEquals(starsBefore - TribesConfig.BRIDGE_COST, scenario.tribes[0].getStars(),
                "Bridge construction should spend five stars.");
    }

    private static void testObservationCopyDoesNotInventBridgeThroughFog() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        scenario.addCapital(0, 1, 2);
        scenario.board.setTerrainAt(2, 1, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setTerrainAt(2, 2, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setTerrainAt(3, 2, Types.TERRAIN.DEEP_WATER);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        GameState authoritative = scenario.buildState();

        scenario.tribes[0].clearView(2, 2, 0, authoritative.getRandomGenerator(), authoritative.getBoard());

        GameState observed = authoritative.copy(0);
        BuildRoad buildBridge = new BuildRoad(0);
        buildBridge.setPosition(new Vector2d(2, 2));

        assertFalse(buildBridge.isFeasible(observed),
                "A player observation copy should not allow a bridge when the unseen far endpoint may still be water.");
    }

    private static void testBridgeCanTargetFoggedFarShoreline() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        scenario.addCapital(0, 1, 3);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.PLAIN);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        GameState gs = scenario.buildState();

        scenario.tribes[0].revealTile(1, 3);
        scenario.tribes[0].revealTile(2, 3);
        assertFalse(scenario.tribes[0].isExplored(3, 3), "Far shoreline should stay unexplored for this bridge check.");

        BuildRoad buildBridge = new BuildRoad(0);
        buildBridge.setPosition(new Vector2d(2, 3));
        assertTrue(buildBridge.isFeasible(gs),
                "Bridge placement should be legal when the visible bank leads to a real land tile in fog.");
    }

    private static void testBuildActionsRequireExploredEmptyCityTiles() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.FOREST);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        GameState gs = scenario.buildState();

        Build lumberHut = new Build(capital.getActorId());
        lumberHut.setBuildingType(Types.BUILDING.LUMBER_HUT);
        lumberHut.setTargetPos(new Vector2d(2, 3));

        assertTrue(lumberHut.isFeasible(gs), "Explored empty city tiles should allow a legal build.");

        scenario.board.setBuildingAt(2, 3, Types.BUILDING.LUMBER_HUT);
        assertFalse(lumberHut.isFeasible(gs), "Builds should be rejected when the target tile already has a building.");

        scenario.board.setBuildingAt(2, 3, null);
        assertTrue(lumberHut.isFeasible(gs), "Builds should remain legal on explored city tiles that are currently fogged.");

        scenario.tribes[0].getObsGrid()[2][3] = false;
        assertFalse(lumberHut.isFeasible(gs), "Builds should be rejected on city tiles that have never been explored.");
        scenario.tribes[0].getObsGrid()[2][3] = true;

        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.SMITHERY);
        scenario.board.setTerrainAt(2, 1, Types.TERRAIN.MOUNTAIN);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.PLAIN);
        scenario.board.setTerrainAt(3, 2, Types.TERRAIN.PLAIN);

        Building mine = new Building(2, 1, Types.BUILDING.MINE, capital.getActorId());
        scenario.board.setBuildingAt(2, 1, Types.BUILDING.MINE);
        capital.addBuilding(gs, mine);

        Building forge = new Building(2, 3, Types.BUILDING.FORGE, capital.getActorId());
        scenario.board.setBuildingAt(2, 3, Types.BUILDING.FORGE);
        capital.addBuilding(gs, forge);
        GameState observed = gs.copy(0);

        Build duplicateForge = new Build(capital.getActorId());
        duplicateForge.setBuildingType(Types.BUILDING.FORGE);
        duplicateForge.setTargetPos(new Vector2d(3, 2));
        assertFalse(duplicateForge.isFeasible(observed),
                "Owned city copies should still reject a duplicate unique building when the first copy is on a hidden tile.");
    }

    private static void testConnectingRoadsAddPopulationToCapitalAndCity() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 3);
        City city = scenario.addCity(0, 5, 3);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        GameState gs = scenario.buildState();

        scenario.tribes[0].clearView(3, 3, 2, gs.getRandomGenerator(), gs.getBoard());
        int capitalPopulationBefore = capital.getPopulation();
        int cityPopulationBefore = city.getPopulation();

        buildRoad(gs, 0, 2, 3);
        buildRoad(gs, 0, 3, 3);
        buildRoad(gs, 0, 4, 3);

        assertTrue(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Connecting the road network should register the city as connected to the capital.");
        assertEquals(capitalPopulationBefore + 1, capital.getPopulation(),
                "Connecting a city should add one population to the capital.");
        assertEquals(cityPopulationBefore + 1, city.getPopulation(),
                "Connecting a city should add one population to the newly connected city.");
        assertTrue(scenario.tribes[0].isMonumentBuildable(Types.BUILDING.GRAND_BAZAR),
                "Connecting the capital to another city should unlock the Network monument.");
    }

    private static void testUnexploredRoadSegmentsBreakCityConnections() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 3);
        City city = scenario.addCity(0, 5, 3);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        GameState gs = scenario.buildState();

        scenario.tribes[0].clearView(3, 3, 2, gs.getRandomGenerator(), gs.getBoard());
        buildRoad(gs, 0, 2, 3);
        buildRoad(gs, 0, 3, 3);
        buildRoad(gs, 0, 4, 3);
        assertTrue(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Setup should connect the city before fog is applied.");

        boolean[][] obsGrid = scenario.tribes[0].getObsGrid();
        obsGrid[3][3] = false;
        obsGrid[4][3] = false;
        gs.getBoard().recomputeTradeNetworkForTribe(0);

        assertFalse(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Unexplored road segments should break city connections.");
        assertFalse(scenario.tribes[0].isExplored(3, 3), "Middle road should be unexplored in this scenario.");
        assertFalse(scenario.tribes[0].isExplored(4, 3), "Far road should be unexplored in this scenario.");
        assertEquals(0, city.getPopulation(), "Disconnecting the city should remove the road-connection population.");
    }

    private static void testAlliedRoadsCanCompleteCityConnections() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City capital = scenario.addCapital(0, 1, 3);
        City city = scenario.addCity(0, 5, 3);
        City allyCity = scenario.addCity(1, 3, 4);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        scenario.tribes[0].setStars(20);
        scenario.tribes[1].setStars(20);
        scenario.board.getDiplomacy().setRelationship(0, 1, Types.RELATIONSHIP.TREATY);
        GameState gs = scenario.buildState();

        scenario.tribes[0].clearView(3, 3, 2, gs.getRandomGenerator(), gs.getBoard());
        buildRoad(gs, 0, 2, 3);
        scenario.board.addRoad(3, 3);
        buildRoad(gs, 0, 4, 3);
        gs.getBoard().recomputeTradeNetworkForTribe(0);

        assertTrue(gs.getBoard().canUseRoad(0, 3, 3), "Treaty allies should be able to use allied roads.");
        assertTrue(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Allied roads should be able to complete a city connection.");
        assertEquals(allyCity.getActorId(), gs.getBoard().getCityIdAt(3, 3),
                "The shared road tile should remain in allied territory for this connection test.");
        assertEquals(1, city.getPopulation(), "Connected city should gain one population through the allied road link.");
        assertEquals(1, capital.getPopulation(), "Capital should gain one population when the allied road link completes.");
    }

    private static void testOceanPortLinksRequireNavigation() {
        Scenario scenario = new Scenario(7, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 3);
        City city = scenario.addCity(0, 5, 3);
        scenario.board.setTerrainAt(2, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(4, 3, Types.TERRAIN.SHALLOW_WATER);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FISHING);
        GameState gs = scenario.buildState();

        scenario.tribes[0].revealTile(3, 3);
        addBuilding(gs, capital, Types.BUILDING.PORT, 2, 3);
        addBuilding(gs, city, Types.BUILDING.PORT, 4, 3);
        gs.getBoard().recomputeTradeNetworkForTribe(0);
        assertFalse(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Port links that cross ocean should not connect cities before Navigation.");

        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.NAVIGATION);
        scenario.tribes[0].revealTile(3, 3);
        gs.getBoard().recomputeTradeNetworkForTribe(0);
        assertTrue(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Port links that cross ocean should connect once Navigation is researched.");
    }

    private static void testPortLinksSpanFiveWaterTiles() {
        Scenario scenario = new Scenario(9, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 0, 4);
        City city = scenario.addCity(0, 8, 4);
        for (int x = 1; x <= 7; x++) {
            scenario.board.setTerrainAt(x, 4, Types.TERRAIN.SHALLOW_WATER);
        }
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FISHING);
        GameState gs = scenario.buildState();

        for (int x = 2; x <= 6; x++) {
            scenario.tribes[0].revealTile(x, 4);
        }
        addBuilding(gs, capital, Types.BUILDING.PORT, 1, 4);
        addBuilding(gs, city, Types.BUILDING.PORT, 7, 4);
        gs.getBoard().recomputeTradeNetworkForTribe(0);

        assertTrue(scenario.tribes[0].getConnectedCities().contains(city.getActorId()),
                "Port links should span five water tiles between ports.");
    }

    private static void testGameStateCopyClonesRandomState() throws Exception {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 0, 0);
        scenario.addCapital(1, 4, 4);
        GameState original = scenario.buildState();
        GameState copy = original.copy();

        Random originalRandom = extractRandom(original);
        Random copyRandom = extractRandom(copy);

        assertTrue(originalRandom != copyRandom, "Copied state should have its own Random instance");
        assertEquals(originalRandom.nextInt(), copyRandom.nextInt(), "Copied state should preserve the RNG sequence");
        assertEquals(originalRandom.nextInt(), copyRandom.nextInt(), "Copied state should preserve the full RNG state");
    }

    private static void testTribeResultTieBreakIsDeterministic() {
        TribeResult left = new TribeResult(1, Types.RESULT.INCOMPLETE, 20, 3, 2, 5);
        TribeResult right = new TribeResult(4, Types.RESULT.INCOMPLETE, 20, 3, 2, 5);

        assertEquals(-1, left.compareTo(right), "Lower tribe id should win a fully tied comparison");
        assertEquals(1, right.compareTo(left), "Higher tribe id should lose a fully tied comparison");
        assertEquals(0, left.compareTo(left.copy()), "Identical tribe results should compare as equal");
    }

    private static void testGameModeCanonicalModes() {
        assertEquals(Types.GAME_MODE.MIGHT, Types.GAME_MODE.CAPITALS.getCanonicalMode(),
                "Legacy Capitals mode should resolve to live Might mode.");
        assertEquals(Types.GAME_MODE.PERFECTION, Types.GAME_MODE.SCORE.getCanonicalMode(),
                "Legacy Score mode should resolve to live Perfection mode.");
        assertEquals(Types.GAME_MODE.SCORE.getMaxTurns(), Types.GAME_MODE.PERFECTION.getMaxTurns(),
                "Perfection should keep the fixed 30-turn cap.");
        assertTrue(Types.GAME_MODE.MIGHT.usesCapitalObjective(),
                "Might should use the capital objective.");
        assertTrue(Types.GAME_MODE.DOMINATION.usesEliminationObjective(),
                "Domination should use the elimination objective.");
        assertTrue(Types.GAME_MODE.GLORY.usesGloryObjective(),
                "Glory should use the score-target objective.");
    }

    private static void testPerfectionEndsOnTurnLimit() {
        Scenario scenario = new Scenario(5, Types.GAME_MODE.PERFECTION, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        scenario.tribes[0].addScore(100);
        GameState gs = scenario.buildState(Types.GAME_MODE.PERFECTION.getMaxTurns());

        assertTrue(gs.gameOver(), "Perfection should end once the fixed turn limit is reached.");
        assertEquals(Types.RESULT.WIN, scenario.tribes[0].getWinner(),
                "Highest-ranked tribe should win when Perfection reaches the turn cap.");
    }

    private static void testTurnLimitOverrideDrawsMightGames() {
        Scenario scenario = new Scenario(5, Types.GAME_MODE.MIGHT, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        scenario.tribes[0].addScore(100);
        GameState gs = scenario.buildState(12);
        gs.setTurnLimitOverride(12);

        assertTrue(gs.gameOver(), "A finite turn-limit override should end otherwise endless Might games.");
        assertEquals(Types.RESULT.INCOMPLETE, scenario.tribes[0].getWinner(),
                "Turn-limited stalled capital games should not force a ranked winner.");
        assertEquals(Types.RESULT.INCOMPLETE, scenario.tribes[1].getWinner(),
                "Turn-limited stalled capital games should leave every non-eliminated tribe drawn.");
    }

    private static void testGloryResolvesAtRoundEnd() {
        Scenario scenario = new Scenario(5, Types.GAME_MODE.GLORY, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        scenario.tribes[0].addScore(5000);
        GameState gs = scenario.buildState(0, 5000);

        assertFalse(gs.gameOver(), "Glory should wait until round end after a tribe reaches the target score.");
        assertEquals(0, gs.getGloryResolutionRound(), "Glory should remember the round in which the target was first reached.");

        gs.incTick();
        assertTrue(gs.gameOver(), "Glory should resolve once the round finishes.");
        assertEquals(Types.RESULT.WIN, scenario.tribes[0].getWinner(),
                "The top-ranked tribe at round end should win Glory.");
        assertEquals(5000, gs.getGloryTargetScore(), "Configured Glory targets should be stored on the game state.");
    }

    private static void testDominationEndsWhenOneTribeRemains() {
        Scenario scenario = new Scenario(5, Types.GAME_MODE.DOMINATION, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        scenario.tribes[1].setWinner(Types.RESULT.LOSS);
        GameState gs = scenario.buildState();

        assertTrue(gs.gameOver(), "Domination should end when only one tribe remains undefeated.");
        assertEquals(Types.RESULT.WIN, scenario.tribes[0].getWinner(),
                "The remaining undefeated tribe should win Domination.");
    }

    private static void testLevelGeneratorRejectsUnsupportedMapSizes() {
        core.levelgen.LevelGenerator levelGenerator = new core.levelgen.LevelGenerator(SCENARIO_SEED);
        try {
            levelGenerator.init(12, Types.MAP_TYPE.CONTINENTS, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR});
            throw new AssertionError("Unsupported map sizes should throw an IllegalArgumentException.");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Unsupported map size"),
                    "Unsupported map-size rejection should explain the failure.");
        }
    }

    private static void testLevelGeneratorKeepsConfirmedCapitalGuarantees() {
        assertCapitalGuarantee(Types.TRIBE.XIN_XI, Types.MAP_SIZE.TINY, Types.MAP_TYPE.CONTINENTS,
                Types.RESOURCE.ORE, Types.TERRAIN.MOUNTAIN, 2, 12001L,
                "Xin-Xi capitals should keep at least two adjacent mountain ore tiles.");
        GeneratedLevel tinyContinents = generateLevel(13L,
                new Types.TRIBE[]{Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS},
                Types.MAP_SIZE.TINY, Types.MAP_TYPE.CONTINENTS);
        assertEquals(2, findAllTerrain(tinyContinents, Types.TERRAIN.CITY).size(),
                "Tiny Continents seed 13 should generate a capital for every player.");
        assertCapitalGuarantee(Types.TRIBE.OUMAJI, Types.MAP_SIZE.TINY, Types.MAP_TYPE.DRYLANDS,
                Types.RESOURCE.FRUIT, Types.TERRAIN.PLAIN, 2, 12002L,
                "Oumaji capitals should keep at least two adjacent fruit tiles.");
    }

    private static void testLevelGeneratorAvoidsShallowWaterRuins() {
        GeneratedLevel level = generateLevel(12004L, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.CONTINENTS);

        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.terrain[x][y] == Types.TERRAIN.SHALLOW_WATER) {
                    assertTrue(level.resource[x][y] != Types.RESOURCE.RUINS,
                            "Current map generation should not place ruins on shallow water.");
                }
            }
        }
    }

    private static void testOceanRuinsStayTwoTilesFromVillages() {
        GeneratedLevel level = generateLevel(12006L, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.KICKOO},
                Types.MAP_SIZE.LARGE, Types.MAP_TYPE.ARCHIPELAGO);

        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.terrain[x][y] == Types.TERRAIN.DEEP_WATER && level.resource[x][y] == Types.RESOURCE.RUINS) {
                    for (int nx = Math.max(0, x - 1); nx <= Math.min(level.size - 1, x + 1); nx++) {
                        for (int ny = Math.max(0, y - 1); ny <= Math.min(level.size - 1, y + 1); ny++) {
                            assertTrue(level.terrain[nx][ny] != Types.TERRAIN.VILLAGE,
                                    "Ocean ruins should not be adjacent to villages.");
                        }
                    }
                }
            }
        }
    }

    private static void testDeepWaterNeverTouchesLandOrthogonally() {
        Types.MAP_TYPE[] mapTypes = new Types.MAP_TYPE[]{
                Types.MAP_TYPE.CONTINENTS,
                Types.MAP_TYPE.DRYLANDS,
                Types.MAP_TYPE.LAKES,
                Types.MAP_TYPE.PANGEA,
                Types.MAP_TYPE.ARCHIPELAGO,
                Types.MAP_TYPE.WATER_WORLD
        };

        long seed = 13000L;
        for (Types.MAP_TYPE mapType : mapTypes) {
            GeneratedLevel level = generateLevel(seed++, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                    Types.MAP_SIZE.NORMAL, mapType);

            for (int x = 0; x < level.size; x++) {
                for (int y = 0; y < level.size; y++) {
                    if (level.terrain[x][y] != Types.TERRAIN.DEEP_WATER) {
                        continue;
                    }
                    Vector2d pos = new Vector2d(x, y);
                    for (Vector2d neighbour : pos.neighborhood(1, 0, level.size)) {
                        if (Math.abs(neighbour.x - x) + Math.abs(neighbour.y - y) != 1) {
                            continue;
                        }
                        Types.TERRAIN adjacent = level.terrain[neighbour.x][neighbour.y];
                        assertTrue(adjacent == Types.TERRAIN.DEEP_WATER || adjacent == Types.TERRAIN.SHALLOW_WATER,
                                "Deep water should only border shallow or deep water orthogonally. Found "
                                        + adjacent + " next to deep water on " + mapType + ".");
                    }
                }
            }
        }
    }

    private static void testMapTypesFollowExpectedWaterProfiles() {
        Types.TRIBE[] tribes = new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.KICKOO};
        GeneratedLevel drylands = generateLevel(13101L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.DRYLANDS);
        GeneratedLevel lakes = generateLevel(13102L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.LAKES);
        GeneratedLevel continents = generateLevel(13103L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.CONTINENTS);
        GeneratedLevel pangea = generateLevel(13104L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.PANGEA);
        GeneratedLevel archipelago = generateLevel(13105L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.ARCHIPELAGO);
        GeneratedLevel waterWorld = generateLevel(13106L, tribes, Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.WATER_WORLD);

        assertTrue(waterRatio(drylands) <= 0.15, "Drylands should be almost entirely land.");
        assertTrue(waterRatio(lakes) >= 0.18 && waterRatio(lakes) <= 0.45,
                "Lakes should have moderate inland water. Found " + waterRatio(lakes) + ".");
        assertTrue(waterRatio(continents) >= 0.30 && waterRatio(continents) <= 0.75,
                "Continents should have substantial water around discrete landmasses.");
        assertTrue(waterRatio(pangea) >= 0.35 && waterRatio(pangea) <= 0.70,
                "Pangea should be roughly half water.");
        assertTrue(waterRatio(archipelago) >= 0.50,
                "Archipelago should be water-heavy.");
        assertTrue(waterRatio(waterWorld) >= 0.75,
                "Water World should be overwhelmingly water.");
        assertTrue(lakesEdgeIsLand(lakes), "Lakes should preserve land bridges around the map edge.");
        assertTrue(largestLandComponentShare(pangea) >= 0.45,
                "Pangea should have one dominant landmass.");
        assertTrue(coastalCityShare(archipelago) >= 0.70,
                "Archipelago should make most cities and villages coastal.");
    }

    private static void testLakesCapitalsKeepLandAccessToTwoVillages() {
        for (long seed = 1L; seed <= 50L; seed++) {
            GeneratedLevel level = generateLevel(seed,
                    new Types.TRIBE[]{Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS},
                    Types.MAP_SIZE.TINY, Types.MAP_TYPE.LAKES);
            ArrayList<Vector2d> capitals = findAllTerrain(level, Types.TERRAIN.CITY);
            assertEquals(2, capitals.size(), "Tiny Lakes should generate one capital per player.");
            for (Vector2d capital : capitals) {
                assertTrue(reachableVillageCount(level, capital) >= 2,
                        "Each Lakes capital should have a land connection to at least two villages. Failed on seed " + seed + ".");
            }
        }
    }

    private static void testLakesVillageSpacingFromCapitalsAndEachOther() {
        for (long seed = 1L; seed <= 50L; seed++) {
            GeneratedLevel level = generateLevel(seed,
                    new Types.TRIBE[]{Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS},
                    Types.MAP_SIZE.TINY, Types.MAP_TYPE.LAKES);
            ArrayList<Vector2d> capitals = findAllTerrain(level, Types.TERRAIN.CITY);
            ArrayList<Vector2d> villages = findAllTerrain(level, Types.TERRAIN.VILLAGE);
            for (Vector2d village : villages) {
                for (Vector2d capital : capitals) {
                    assertTrue(distance(village, capital) >= 2,
                            "Lakes villages should stay at least two tiles from capitals. Failed on seed " + seed + ".");
                }
            }
            for (int i = 0; i < villages.size(); i++) {
                for (int j = i + 1; j < villages.size(); j++) {
                    assertTrue(distance(villages.get(i), villages.get(j)) >= 2,
                            "Lakes villages should stay at least two tiles from each other. Failed on seed " + seed + ".");
                }
            }
        }
    }

    private static void testContinentsPlaceCapitalsOnSeparateLandmasses() {
        GeneratedLevel level = generateLevel(13120L,
                new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI, Types.TRIBE.OUMAJI},
                Types.MAP_SIZE.LARGE, Types.MAP_TYPE.CONTINENTS);
        HashMap<Integer, Boolean> capitalComponents = new HashMap<>();
        int capitals = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.terrain[x][y] == Types.TERRAIN.CITY) {
                    capitals++;
                    capitalComponents.put(landComponentKey(level, x, y), true);
                }
            }
        }
        assertEquals(capitals, capitalComponents.size(),
                "Continents should place capitals on separate landmasses when enough landmasses exist.");
    }

    private static void testContinentsLimitOneTileIslandVillages() {
        GeneratedLevel level = generateLevel(13121L,
                new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI},
                Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.CONTINENTS);
        int isolatedCityTiles = countIsolatedOneTileCityComponents(level);
        assertTrue(isolatedCityTiles <= 2,
                "Normal Continents should only have the two configured tiny-island villages as one-tile city islands.");
    }

    private static void testContinentsAllowEdgeLandWithoutWalkableContinentContact() {
        GeneratedLevel level = generateLevel(13122L,
                new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI},
                Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.CONTINENTS);
        assertTrue(hasEdgeLand(level), "Continents should allow landmasses to naturally reach map edges.");
        assertTrue(separateLandmassesDoNotTouchDiagonally(level),
                "Separate Continents landmasses should not touch cardinally or diagonally.");
    }

    private static void testContinentsKeepCapitalsSpacedAndGiveEachLandmassVillages() {
        GeneratedLevel level = generateLevel(13123L,
                new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                Types.MAP_SIZE.TINY, Types.MAP_TYPE.CONTINENTS);
        ArrayList<Vector2d> capitals = findAllTerrain(level, Types.TERRAIN.CITY);
        for (int i = 0; i < capitals.size(); i++) {
            for (int j = i + 1; j < capitals.size(); j++) {
                assertTrue(distance(capitals.get(i), capitals.get(j)) >= 4,
                        "Continents capitals should not spawn close enough to see each other immediately.");
            }
        }
        for (Vector2d capital : capitals) {
            int component = landComponentKey(level, capital.x, capital.y);
            assertTrue(componentHasVillage(level, component),
                    "Each starting Continents landmass should have at least one non-capital village when there is room.");
        }
    }

    private static void testGeneratedLandResourcesStayNearCities() {
        for (Types.MAP_TYPE mapType : Types.MAP_TYPE.values()) {
            GeneratedLevel level = generateLevel(13200L + mapType.ordinal(),
                    new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI},
                    Types.MAP_SIZE.NORMAL, mapType);
            for (int x = 0; x < level.size; x++) {
                for (int y = 0; y < level.size; y++) {
                    Types.RESOURCE resource = level.resource[x][y];
                    if (resource == Types.RESOURCE.FRUIT || resource == Types.RESOURCE.CROPS
                            || resource == Types.RESOURCE.ANIMAL || resource == Types.RESOURCE.ORE) {
                        assertTrue(hasCityOrVillageWithin(level, x, y, 2),
                                "Generated land resources should stay within two tiles of cities or villages.");
                    }
                }
            }
        }
    }

    private static void testGeneratedRuinsObeyTerrainAndSpacing() {
        for (Types.MAP_TYPE mapType : Types.MAP_TYPE.values()) {
            GeneratedLevel level = generateLevel(13300L + mapType.ordinal(),
                    new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.OUMAJI},
                    Types.MAP_SIZE.NORMAL, mapType);
            int waterRuins = 0;
            int totalRuins = 0;
            for (int x = 0; x < level.size; x++) {
                for (int y = 0; y < level.size; y++) {
                    if (level.resource[x][y] != Types.RESOURCE.RUINS) {
                        continue;
                    }
                    totalRuins++;
                    Types.TERRAIN terrain = level.terrain[x][y];
                    assertTrue(terrain == Types.TERRAIN.PLAIN || terrain == Types.TERRAIN.FOREST
                                    || terrain == Types.TERRAIN.MOUNTAIN || terrain == Types.TERRAIN.DEEP_WATER,
                            "Ruins should only spawn on field, forest, mountain, or deep water. Found " + terrain
                                    + " on " + mapType + " at " + x + "," + y + ".");
                    if (terrain == Types.TERRAIN.DEEP_WATER) {
                        waterRuins++;
                    }
                    for (Vector2d near : new Vector2d(x, y).neighborhood(1, 0, level.size)) {
                        if (near.x == x && near.y == y) {
                            continue;
                        }
                        assertTrue(level.resource[near.x][near.y] != Types.RESOURCE.RUINS,
                                "Ruins should not spawn adjacent to other ruins.");
                        assertTrue(level.terrain[near.x][near.y] != Types.TERRAIN.VILLAGE
                                        && level.terrain[near.x][near.y] != Types.TERRAIN.CITY,
                                "Ruins should not spawn adjacent to cities or villages.");
                    }
                }
            }
            assertEquals(7, totalRuins, "Normal maps should generate seven ruins.");
            if (mapType == Types.MAP_TYPE.LAKES) {
                assertTrue(waterRuins <= 2, "Lakes should cap water ruins at one third of total ruins.");
            }
        }
    }

    private static void testStarfishDensityFollowsWaterTileCount() {
        for (Types.MAP_TYPE mapType : Types.MAP_TYPE.values()) {
            GeneratedLevel level = generateLevel(13400L + mapType.ordinal(),
                    new Types.TRIBE[]{Types.TRIBE.KICKOO, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                    Types.MAP_SIZE.NORMAL, mapType);
            int starfish = countResource(level, Types.RESOURCE.STARFISH);
            int eligibleWater = countEmptyWaterTiles(level) + starfish;
            int expected = eligibleWater / 25;
            assertTrue(starfish <= expected && starfish >= Math.max(0, expected - 2),
                    "Starfish count should track roughly one per 25 eligible water tiles. Expected around "
                            + expected + " and found " + starfish + " on " + mapType + ".");
        }
    }

    private static void testLevelGeneratorPlacesCornerLighthouses() {
        GeneratedLevel level = generateLevel(12003L, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                Types.MAP_SIZE.TINY, Types.MAP_TYPE.CONTINENTS);

        int max = level.size - 1;
        assertEquals(Types.RESOURCE.LIGHTHOUSE, level.resource[0][0],
                "Top-left corner should contain a lighthouse.");
        assertEquals(Types.RESOURCE.LIGHTHOUSE, level.resource[0][max],
                "Top-right corner should contain a lighthouse.");
        assertEquals(Types.RESOURCE.LIGHTHOUSE, level.resource[max][0],
                "Bottom-left corner should contain a lighthouse.");
        assertEquals(Types.RESOURCE.LIGHTHOUSE, level.resource[max][max],
                "Bottom-right corner should contain a lighthouse.");

        int[][] corners = new int[][]{{0, 0}, {0, max}, {max, 0}, {max, max}};
        for (int[] corner : corners) {
            Types.TERRAIN terrain = level.terrain[corner[0]][corner[1]];
            assertTrue(terrain != Types.TERRAIN.MOUNTAIN,
                    "Generated lighthouses should not be placed on mountains.");
            assertTrue(terrain != Types.TERRAIN.FOREST,
                    "Generated lighthouses should not be placed on forests.");
        }
    }

    private static void testLevelGeneratorUsesLiveRuinCounts() {
        assertGeneratedRuinCount(Types.MAP_SIZE.TINY, 4, 22001L);
        assertGeneratedRuinCount(Types.MAP_SIZE.SMALL, 5, 22002L);
        assertGeneratedRuinCount(Types.MAP_SIZE.NORMAL, 7, 22003L);
        assertGeneratedRuinCount(Types.MAP_SIZE.LARGE, 9, 22004L);
        assertGeneratedRuinCount(Types.MAP_SIZE.HUGE, 11, 22005L);
        assertGeneratedRuinCount(Types.MAP_SIZE.MASSIVE, 23, 22006L);
    }

    private static void testLevelGeneratorKeepsRuinsOffLighthouseCorners() {
        GeneratedLevel level = generateLevel(22007L, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.CONTINENTS);

        int max = level.size - 1;
        int[][] corners = new int[][]{{0, 0}, {0, max}, {max, 0}, {max, max}};
        for (int[] corner : corners) {
            assertEquals(Types.RESOURCE.LIGHTHOUSE, level.resource[corner[0]][corner[1]],
                    "Corner lighthouse placement should happen before ruins and remain intact.");
        }
    }

    private static void testLevelGeneratorSpacesStarfish() {
        GeneratedLevel level = generateLevel(22008L,
                new Types.TRIBE[]{Types.TRIBE.KICKOO, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                Types.MAP_SIZE.NORMAL, Types.MAP_TYPE.ARCHIPELAGO);

        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.resource[x][y] != Types.RESOURCE.STARFISH) {
                    continue;
                }
                for (Vector2d adjacent : new Vector2d(x, y).neighborhood(1, 0, level.size)) {
                    assertTrue(level.resource[adjacent.x][adjacent.y] != Types.RESOURCE.STARFISH,
                            "Generated starfish should not be adjacent to another starfish.");
                    assertTrue(level.resource[adjacent.x][adjacent.y] != Types.RESOURCE.LIGHTHOUSE,
                            "Generated starfish should not be adjacent to a lighthouse.");
                    assertTrue(level.terrain[adjacent.x][adjacent.y] != Types.TERRAIN.CITY,
                            "Generated starfish should not be adjacent to a city.");
                }
            }
        }
    }

    private static void testLighthouseDiscoveryUnlocksEyeOfGod() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        Board board = gs.getBoard();
        int[][] corners = new int[][]{{0, 0}, {0, 4}, {4, 0}, {4, 4}};

        for (int[] corner : corners) {
            board.setTerrainAt(corner[0], corner[1], Types.TERRAIN.SHALLOW_WATER);
            board.setResourceAt(corner[0], corner[1], Types.RESOURCE.LIGHTHOUSE);
        }

        int initialPopulation = capital.getPopulation();
        for (int[] corner : corners) {
            scenario.tribes[0].clearView(corner[0], corner[1], 0, gs.getRandomGenerator(), board);
        }

        assertEquals(initialPopulation + 4, capital.getPopulation(),
                "Each newly discovered lighthouse should add one population to the capital.");
        assertTrue(scenario.tribes[0].isMonumentBuildable(Types.BUILDING.EYE_OF_GOD),
                "Discovering all four lighthouses should unlock Eye of God.");
    }

    private static void testLighthouseDiscoveryRewardsEachLighthouseOnce() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 2, 2);
        GameState gs = scenario.buildState();
        Board board = gs.getBoard();

        board.setTerrainAt(4, 4, Types.TERRAIN.SHALLOW_WATER);
        board.setResourceAt(4, 4, Types.RESOURCE.LIGHTHOUSE);
        int populationBefore = capital.getPopulation();

        scenario.tribes[0].clearView(4, 4, 0, gs.getRandomGenerator(), board);
        scenario.tribes[0].clearView(4, 4, 0, gs.getRandomGenerator(), board);

        assertEquals(populationBefore + 1, capital.getPopulation(),
                "Rediscovering the same lighthouse should not grant population twice.");
        assertEquals(1, scenario.tribes[0].getDiscoveredLighthouses().size(),
                "Each lighthouse should only be recorded once per tribe.");
    }

    private static void assertCapitalGuarantee(Types.TRIBE tribeType, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType,
                                               Types.RESOURCE resource, Types.TERRAIN terrain, int minimumCount,
                                               long seed, String failureMessage) {
        GeneratedLevel level = generateLevel(seed, new Types.TRIBE[]{tribeType, Types.TRIBE.IMPERIUS}, mapSize, mapType);
        Vector2d capital = findCapital(level, 0, tribeType.getKey());
        int matchingTiles = countAdjacentTerrainResource(level, capital, terrain, resource);

        assertTrue(level.size == mapSize.getSideLength(),
                "Generated board should use the requested supported map size.");
        assertTrue(matchingTiles >= minimumCount, failureMessage + " Found " + matchingTiles + ".");
    }

    private static int countAdjacentTerrainResource(GeneratedLevel level, Vector2d center, Types.TERRAIN terrain, Types.RESOURCE resource) {
        int matches = 0;
        for (Vector2d adjacent : center.neighborhood(1, 0, level.size)) {
            if (level.terrain[adjacent.x][adjacent.y] == terrain
                    && level.resource[adjacent.x][adjacent.y] == resource) {
                matches++;
            }
        }
        return matches;
    }

    private static void assertGeneratedRuinCount(Types.MAP_SIZE mapSize, int expectedRuinCount, long seed) {
        GeneratedLevel level = generateLevel(seed, new Types.TRIBE[]{Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR},
                mapSize, Types.MAP_TYPE.CONTINENTS);
        int ruinCount = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.resource[x][y] == Types.RESOURCE.RUINS) {
                    ruinCount++;
                }
            }
        }

        assertEquals(expectedRuinCount, ruinCount,
                "Generated ruin count should match the live Polytopia table for " + mapSize + ".");
    }

    private static double waterRatio(GeneratedLevel level) {
        return countWaterTiles(level) / (double) (level.size * level.size);
    }

    private static int countWaterTiles(GeneratedLevel level) {
        int count = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (isWater(level.terrain[x][y])) {
                    count++;
                }
            }
        }
        return count;
    }

    private static int countResource(GeneratedLevel level, Types.RESOURCE resource) {
        int count = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.resource[x][y] == resource) {
                    count++;
                }
            }
        }
        return count;
    }

    private static int countEmptyWaterTiles(GeneratedLevel level) {
        int count = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (isWater(level.terrain[x][y]) && level.resource[x][y] == null) {
                    count++;
                }
            }
        }
        return count;
    }

    private static boolean isWater(Types.TERRAIN terrain) {
        return terrain == Types.TERRAIN.SHALLOW_WATER || terrain == Types.TERRAIN.DEEP_WATER;
    }

    private static boolean lakesEdgeIsLand(GeneratedLevel level) {
        int max = level.size - 1;
        for (int i = 0; i < level.size; i++) {
            if (isWater(level.terrain[0][i]) || isWater(level.terrain[max][i])
                    || isWater(level.terrain[i][0]) || isWater(level.terrain[i][max])) {
                return false;
            }
        }
        return true;
    }

    private static boolean hasCityOrVillageWithin(GeneratedLevel level, int x, int y, int radius) {
        for (int nx = Math.max(0, x - radius); nx <= Math.min(level.size - 1, x + radius); nx++) {
            for (int ny = Math.max(0, y - radius); ny <= Math.min(level.size - 1, y + radius); ny++) {
                Types.TERRAIN terrain = level.terrain[nx][ny];
                if (terrain == Types.TERRAIN.CITY || terrain == Types.TERRAIN.VILLAGE) {
                    return true;
                }
            }
        }
        return false;
    }

    private static double coastalCityShare(GeneratedLevel level) {
        int total = 0;
        int coastal = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                Types.TERRAIN terrain = level.terrain[x][y];
                if (terrain == Types.TERRAIN.CITY || terrain == Types.TERRAIN.VILLAGE) {
                    total++;
                    if (hasAdjacentWater(level, x, y)) {
                        coastal++;
                    }
                }
            }
        }
        return total == 0 ? 0.0 : coastal / (double) total;
    }

    private static boolean hasAdjacentWater(GeneratedLevel level, int x, int y) {
        int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx >= 0 && ny >= 0 && nx < level.size && ny < level.size && isWater(level.terrain[nx][ny])) {
                return true;
            }
        }
        return false;
    }

    private static ArrayList<Vector2d> findAllTerrain(GeneratedLevel level, Types.TERRAIN terrain) {
        ArrayList<Vector2d> matches = new ArrayList<>();
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.terrain[x][y] == terrain) {
                    matches.add(new Vector2d(x, y));
                }
            }
        }
        return matches;
    }

    private static boolean componentHasVillage(GeneratedLevel level, int component) {
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (level.terrain[x][y] == Types.TERRAIN.VILLAGE && landComponentKey(level, x, y) == component) {
                    return true;
                }
            }
        }
        return false;
    }

    private static ArrayList<Vector2d> crossNeighbors(GeneratedLevel level, int x, int y) {
        ArrayList<Vector2d> neighbors = new ArrayList<>();
        if (x > 0) {
            neighbors.add(new Vector2d(x - 1, y));
        }
        if (x < level.size - 1) {
            neighbors.add(new Vector2d(x + 1, y));
        }
        if (y > 0) {
            neighbors.add(new Vector2d(x, y - 1));
        }
        if (y < level.size - 1) {
            neighbors.add(new Vector2d(x, y + 1));
        }
        return neighbors;
    }

    private static int reachableVillageCount(GeneratedLevel level, Vector2d capital) {
        boolean[][] visited = new boolean[level.size][level.size];
        ArrayDeque<Vector2d> queue = new ArrayDeque<>();
        queue.add(capital);
        visited[capital.x][capital.y] = true;
        int villages = 0;
        while (!queue.isEmpty()) {
            Vector2d cell = queue.removeFirst();
            if (level.terrain[cell.x][cell.y] == Types.TERRAIN.VILLAGE) {
                villages++;
            }
            for (Vector2d next : crossNeighbors(level, cell.x, cell.y)) {
                if (visited[next.x][next.y] || isWater(level.terrain[next.x][next.y])) {
                    continue;
                }
                visited[next.x][next.y] = true;
                queue.add(next);
            }
        }
        return villages;
    }

    private static int distance(Vector2d a, Vector2d b) {
        return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
    }

    private static boolean hasEdgeLand(GeneratedLevel level) {
        int max = level.size - 1;
        for (int i = 0; i < level.size; i++) {
            if (!isWater(level.terrain[0][i]) || !isWater(level.terrain[max][i])
                    || !isWater(level.terrain[i][0]) || !isWater(level.terrain[i][max])) {
                return true;
            }
        }
        return false;
    }

    private static boolean separateLandmassesDoNotTouchDiagonally(GeneratedLevel level) {
        int[][] component = landComponentMap(level);
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (component[x][y] < 0) {
                    continue;
                }
                for (int nx = Math.max(0, x - 1); nx <= Math.min(level.size - 1, x + 1); nx++) {
                    for (int ny = Math.max(0, y - 1); ny <= Math.min(level.size - 1, y + 1); ny++) {
                        if (component[nx][ny] >= 0 && component[nx][ny] != component[x][y]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    private static int[][] landComponentMap(GeneratedLevel level) {
        int[][] component = new int[level.size][level.size];
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                component[x][y] = -1;
            }
        }
        boolean[][] visited = new boolean[level.size][level.size];
        int nextComponent = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (visited[x][y] || isWater(level.terrain[x][y])) {
                    continue;
                }
                ArrayList<Vector2d> tiles = landComponentTiles(level, x, y, visited);
                for (Vector2d tile : tiles) {
                    component[tile.x][tile.y] = nextComponent;
                }
                nextComponent++;
            }
        }
        return component;
    }

    private static int countIsolatedOneTileCityComponents(GeneratedLevel level) {
        int count = 0;
        boolean[][] visited = new boolean[level.size][level.size];
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (visited[x][y] || isWater(level.terrain[x][y])) {
                    continue;
                }
                ArrayList<Vector2d> component = landComponentTiles(level, x, y, visited);
                if (component.size() == 1) {
                    Vector2d tile = component.get(0);
                    Types.TERRAIN terrain = level.terrain[tile.x][tile.y];
                    if (terrain == Types.TERRAIN.CITY || terrain == Types.TERRAIN.VILLAGE) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    private static int landComponentKey(GeneratedLevel level, int x, int y) {
        boolean[][] visited = new boolean[level.size][level.size];
        ArrayList<Vector2d> component = landComponentTiles(level, x, y, visited);
        int key = Integer.MAX_VALUE;
        for (Vector2d tile : component) {
            key = Math.min(key, tile.x * level.size + tile.y);
        }
        return key;
    }

    private static ArrayList<Vector2d> landComponentTiles(GeneratedLevel level, int startX, int startY, boolean[][] visited) {
        ArrayList<Vector2d> component = new ArrayList<>();
        ArrayDeque<Vector2d> queue = new ArrayDeque<>();
        queue.add(new Vector2d(startX, startY));
        visited[startX][startY] = true;
        while (!queue.isEmpty()) {
            Vector2d pos = queue.removeFirst();
            component.add(pos);
            for (int nx = Math.max(0, pos.x - 1); nx <= Math.min(level.size - 1, pos.x + 1); nx++) {
                for (int ny = Math.max(0, pos.y - 1); ny <= Math.min(level.size - 1, pos.y + 1); ny++) {
                    if (nx == pos.x && ny == pos.y) {
                        continue;
                    }
                if (nx >= 0 && ny >= 0 && nx < level.size && ny < level.size
                        && !visited[nx][ny] && !isWater(level.terrain[nx][ny])) {
                    visited[nx][ny] = true;
                    queue.add(new Vector2d(nx, ny));
                }
                }
            }
        }
        return component;
    }

    private static double largestLandComponentShare(GeneratedLevel level) {
        boolean[][] visited = new boolean[level.size][level.size];
        int landTiles = 0;
        int largest = 0;
        for (int x = 0; x < level.size; x++) {
            for (int y = 0; y < level.size; y++) {
                if (isWater(level.terrain[x][y])) {
                    continue;
                }
                landTiles++;
                if (!visited[x][y]) {
                    largest = Math.max(largest, floodLandComponent(level, x, y, visited));
                }
            }
        }
        return landTiles == 0 ? 0.0 : largest / (double) landTiles;
    }

    private static int floodLandComponent(GeneratedLevel level, int startX, int startY, boolean[][] visited) {
        int count = 0;
        ArrayDeque<Vector2d> queue = new ArrayDeque<>();
        queue.add(new Vector2d(startX, startY));
        visited[startX][startY] = true;
        int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (!queue.isEmpty()) {
            Vector2d pos = queue.removeFirst();
            count++;
            for (int[] dir : dirs) {
                int nx = pos.x + dir[0];
                int ny = pos.y + dir[1];
                if (nx >= 0 && ny >= 0 && nx < level.size && ny < level.size
                        && !visited[nx][ny] && !isWater(level.terrain[nx][ny])) {
                    visited[nx][ny] = true;
                    queue.add(new Vector2d(nx, ny));
                }
            }
        }
        return count;
    }

    private static GeneratedLevel generateLevel(long seed, Types.TRIBE[] tribes, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType) {
        LevelGenerator generator = new LevelGenerator(seed);
        generator.init(mapSize.getSideLength(), mapType, tribes);
        generator.generate();
        return parseGeneratedLevel(generator.gelLevelLines(), mapSize.getSideLength());
    }

    private static GeneratedLevel parseGeneratedLevel(String[] lines, int size) {
        Types.TERRAIN[][] terrain = new Types.TERRAIN[size][size];
        Types.RESOURCE[][] resource = new Types.RESOURCE[size][size];

        for (int row = 0; row < size; row++) {
            String[] tiles = lines[row].split(",");
            for (int col = 0; col < size; col++) {
                String[] pieces = tiles[col].split(":");
                terrain[row][col] = Types.TERRAIN.getType(pieces[0].charAt(0));
                resource[row][col] = parseGeneratedResource(pieces);
            }
        }

        return new GeneratedLevel(size, terrain, resource, lines);
    }

    private static Types.RESOURCE parseGeneratedResource(String[] pieces) {
        if (pieces.length < 2 || pieces[1].isEmpty() || pieces[1].charAt(0) == ' ') {
            return null;
        }
        char resourceChar = pieces[1].charAt(0);
        if (Character.isDigit(resourceChar) || resourceChar == '-') {
            return null;
        }
        return Types.RESOURCE.getType(resourceChar);
    }

    private static Vector2d findCapital(GeneratedLevel level, int playerSlot, int tribeKey) {
        String capitalSuffix = ":" + tribeKey + ":" + playerSlot;
        for (int row = 0; row < level.size; row++) {
            for (int col = 0; col < level.size; col++) {
                if (level.raw[row][col].startsWith("c") && level.raw[row][col].endsWith(capitalSuffix)) {
                    return new Vector2d(row, col);
                }
            }
        }
        throw new AssertionError("Generated level should contain a capital for player slot " + playerSlot + ".");
    }

    private static int countMatchingTests(HarnessSuite suite, String testFilter) {
        if (testFilter == null) {
            return suite.tests.length;
        }
        int matches = 0;
        for (HarnessTest test : suite.tests) {
            if (test.name.toLowerCase().contains(testFilter.toLowerCase())) {
                matches++;
            }
        }
        return matches;
    }

    private static String parseFlagValue(String[] args, String flag) {
        for (int i = 0; i < args.length - 1; i++) {
            if (flag.equals(args[i])) {
                return args[i + 1];
            }
        }
        return null;
    }

    private static String formatTiming(long elapsedMs) {
        if (elapsedMs >= SLOW_TEST_MS) {
            return " (" + elapsedMs + " ms)";
        }
        return "";
    }

    private static void testCombatRetaliation() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCity = scenario.addCapital(0, 0, 0);
        City defenderCity = scenario.addCapital(1, 4, 4);
        Unit attacker = scenario.addUnit(attackerCity, Types.UNIT.WARRIOR, 2, 2);
        Unit defender = scenario.addUnit(defenderCity, Types.UNIT.WARRIOR, 2, 3);
        GameState gs = scenario.buildState();

        attacker.setStatus(Types.TURN_STATUS.FRESH);
        defender.setStatus(Types.TURN_STATUS.FRESH);

        Attack action = new Attack(attacker.getActorId());
        action.setTargetId(defender.getActorId());
        AttackCommand command = new AttackCommand();

        assertTrue(action.isFeasible(gs), "Attack should be feasible for adjacent fresh warriors");
        assertTrue(command.isRetaliation(action, gs), "Defender should be able to retaliate at melee range");
        assertTrue(command.execute(action, gs), "Attack command should execute");

        assertEquals(5, attacker.getCurrentHP(), "Attacker should take retaliation damage");
        assertEquals(5, defender.getCurrentHP(), "Defender should take attack damage");
        assertEquals(Types.TURN_STATUS.FINISHED, attacker.getStatus(), "Warrior should finish its turn after attacking");
        assertEquals(attacker.getActorId(), gs.getBoard().getUnitAt(2, 2).getActorId(), "Attacker should remain in place when the target survives");
        assertEquals(defender.getActorId(), gs.getBoard().getUnitAt(2, 3).getActorId(), "Defender should remain in place when it survives");
    }

    private static void testSplashDamagePreservesHalfHitPoints() {
        Scenario scenario = new Scenario(7, Types.TRIBE.KICKOO, Types.TRIBE.BARDUR);
        City attackerCity = scenario.addCapital(0, 0, 0);
        City defenderCity = scenario.addCapital(1, 6, 6);
        scenario.board.setTerrainAt(3, 3, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(3, 5, Types.TERRAIN.DEEP_WATER);
        scenario.board.setTerrainAt(4, 5, Types.TERRAIN.DEEP_WATER);
        Unit bomber = scenario.addUnit(attackerCity, Types.UNIT.BOMBER, 3, 3);
        Unit target = scenario.addUnit(defenderCity, Types.UNIT.DEFENDER, 3, 5);
        Unit splashTarget = scenario.addUnit(defenderCity, Types.UNIT.DEFENDER, 4, 5);
        GameState gs = scenario.buildState();

        bomber.setStatus(Types.TURN_STATUS.FRESH);
        target.setStatus(Types.TURN_STATUS.FRESH);
        splashTarget.setStatus(Types.TURN_STATUS.FRESH);
        scenario.tribes[0].clearView(3, 5, 1, gs.getRandomGenerator(), gs.getBoard());

        Attack action = new Attack(bomber.getActorId());
        action.setTargetId(target.getActorId());
        assertTrue(action.isFeasible(gs), "Bomber should be able to attack the target defender.");
        assertTrue(new AttackCommand().execute(action, gs), "Bomber attack should execute.");

        assertEquals(11.5, splashTarget.getCurrentHPExact(), 0.0001,
                "Splash damage should divide odd attack damage by two without rounding.");
    }

    private static void testCityCapture() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City attackerCapital = scenario.addCapital(0, 0, 0);
        City defenderCapital = scenario.addCapital(1, 2, 2);
        Unit attacker = scenario.addUnit(attackerCapital, Types.UNIT.WARRIOR, 2, 2);
        GameState gs = scenario.buildState();

        attacker.setStatus(Types.TURN_STATUS.FRESH);

        Capture action = new Capture(attacker.getActorId());
        action.setCaptureType(Types.TERRAIN.CITY);
        action.setTargetCity(defenderCapital.getActorId());

        assertTrue(action.isFeasible(gs), "Capture should be feasible when a fresh unit stands on an enemy city");
        assertTrue(new CaptureCommand().execute(action, gs), "Capture command should execute");

        assertEquals(0, defenderCapital.getTribeId(), "Captured city should change owner");
        assertEquals(2, scenario.tribes[0].getNumCities(), "Capturing tribe should gain the city");
        assertEquals(0, scenario.tribes[1].getNumCities(), "Defending tribe should lose its final city");
        assertEquals(Types.RESULT.LOSS, scenario.tribes[1].getWinner(), "Defending tribe should be marked as defeated");
        assertEquals(Types.TURN_STATUS.FINISHED, attacker.getStatus(), "Capturing unit should exhaust its turn");
        assertEquals(defenderCapital.getActorId(), attacker.getCityId(), "Capturing unit should be reassigned to the captured city");
        assertTrue(defenderCapital.getUnitsID().contains(attacker.getActorId()), "Captured city should now list the capturing unit");
        assertFalse(attackerCapital.getUnitsID().contains(attacker.getActorId()), "Original city should no longer list the capturing unit");
    }

    private static void testDuplicateTribeSeatsPreservePlayerAssignment() throws Exception {
        ArrayList<Agent> seatPlayers = new ArrayList<>();
        seatPlayers.add(new SeatTrackingAgent(100L, "seat-0"));
        seatPlayers.add(new SeatTrackingAgent(200L, "seat-1"));
        seatPlayers.add(new SeatTrackingAgent(300L, "seat-2"));

        Types.TRIBE[] tribes = new Types.TRIBE[]{
                Types.TRIBE.IMPERIUS,
                Types.TRIBE.IMPERIUS,
                Types.TRIBE.XIN_XI
        };

        Game game = new Game();
        game.init(seatPlayers, 42098L, tribes, 99L, Types.GAME_MODE.MIGHT, Types.MAP_SIZE.TINY, Types.MAP_TYPE.LAKES,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);

        Agent[] assignedPlayers = extractPlayers(game);
        assertSame(seatPlayers.get(0), assignedPlayers[0], "Seat 0 should keep its original player even with duplicate tribes.");
        assertSame(seatPlayers.get(1), assignedPlayers[1], "Seat 1 should keep its original player even with duplicate tribes.");
        assertSame(seatPlayers.get(2), assignedPlayers[2], "Seat 2 should keep its original player even with duplicate tribes.");
    }

    private static void testExternalActionRequestUsesCompactSchema() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City capital = scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 1, 2);
        GameState gs = scenario.buildState();
        gs.computePlayerActions(gs.getActiveTribe());

        ArrayList<Action> actions = gs.getAllAvailableActions();
        ArrayList<String> actionIds = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            actionIds.add("A" + i);
        }

        JSONObject payload = ExternalBotPayloadBuilder.buildActionRequest(gs, 0, 5000L, 7, actions, actionIds, "root", true);
        assertFalse(payload.has("protocol_version"), "Compact payload should not include a protocol version.");
        assertTrue(payload.has("obs"), "Compact payload should expose observation under obs.");
        assertFalse(payload.has("observation"), "Verbose observation key should be removed.");
        assertTrue(payload.has("actions"), "Compact payload should expose actions.");
        assertFalse(payload.has("forward_model"), "Verbose forward-model key should be removed.");
        assertTrue(payload.has("fm"), "Compact payload should expose forward-model metadata under fm.");
        assertEquals(5000L, payload.getLong("time_ms"), "Remaining time should use time_ms.");

        JSONObject obs = payload.getJSONObject("obs");
        assertTrue(obs.has("mode"), "Observation should use compact mode field.");
        assertTrue(obs.has("map"), "Observation should expose compact map field.");
        assertEquals(Types.MAP_TYPE.CONTINENTS.name(), obs.getString("map"),
                "Observation should expose the current map type to external engines.");
        assertFalse(obs.has("game_mode"), "Verbose game_mode field should be removed.");
        assertTrue(obs.has("board"), "Observation should include board.");
        assertTrue(obs.has("tribes"), "Observation should include tribes.");
        assertTrue(obs.has("cities"), "Observation should include cities.");
        assertTrue(obs.has("units"), "Observation should include units.");

        JSONObject compactUnit = findEntityById(obs.getJSONArray("units"), warrior.getActorId());
        assertTrue(compactUnit.has("p"), "Units should use compact owner key p.");
        assertTrue(compactUnit.has("hp"), "Units should use compact health key hp.");
        assertFalse(compactUnit.has("tribe_id"), "Verbose unit owner key should be removed.");
        assertFalse(compactUnit.has("current_hp"), "Verbose unit health key should be removed.");

        JSONObject compactCity = findEntityById(obs.getJSONArray("cities"), capital.getActorId());
        assertTrue(compactCity.has("lvl"), "Cities should use compact level key lvl.");
        assertTrue(compactCity.has("b"), "Cities should use compact building list key b.");
        assertFalse(compactCity.has("level"), "Verbose city level key should be removed.");
        assertFalse(compactCity.has("buildings"), "Verbose city buildings key should be removed.");

        JSONObject compactTribe = findEntityById(obs.getJSONArray("tribes"), 0);
        assertTrue(compactTribe.has("tribe"), "Tribes should use compact tribe name key.");
        assertTrue(compactTribe.has("tech"), "Tribes should use compact tech list key.");
        assertFalse(compactTribe.has("name"), "Verbose tribe name key should be removed.");
        assertFalse(compactTribe.has("researched_tech_ids"), "Verbose tribe tech key should be removed.");

        JSONObject firstAction = payload.getJSONArray("actions").getJSONObject(0);
        assertTrue(firstAction.has("i"), "Actions should expose compact action index key i.");
        assertTrue(firstAction.has("t"), "Actions should expose compact action type key t.");
        assertFalse(firstAction.has("index"), "Verbose action index key should be removed.");
        assertFalse(firstAction.has("type"), "Verbose action type key should be removed.");
    }

    private static void testExternalPayloadIncludesHiddenHintsAndLighthouseDiscoverers() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR, Types.TRIBE.XIN_XI);
        City capital = scenario.addCapital(0, 1, 1);
        City bardurCapital = scenario.addCapital(1, 4, 4);
        scenario.addCapital(2, 0, 4);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 2, 2);
        Unit hiddenCloak = scenario.addUnit(bardurCapital, Types.UNIT.CLOAK, 3, 3);
        scenario.board.setResourceAt(4, 0, Types.RESOURCE.LIGHTHOUSE);
        GameState authoritative = scenario.buildState();

        hiddenCloak.setHidden(true);
        scenario.tribes[0].revealArea(authoritative.getBoard(), 4, 0, 0);
        scenario.tribes[1].clearView(4, 0, 0, authoritative.getRandomGenerator(), authoritative.getBoard());
        scenario.tribes[2].clearView(4, 0, 0, authoritative.getRandomGenerator(), authoritative.getBoard());

        authoritative.computePlayerActions(authoritative.getActiveTribe());

        ArrayList<Action> actions = authoritative.getAllAvailableActions();
        ArrayList<String> actionIds = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            actionIds.add("A" + i);
        }

        JSONObject payload = ExternalBotPayloadBuilder.buildActionRequest(authoritative, 0, 5000L, 8, actions, actionIds, "root", true);
        JSONObject obs = payload.getJSONObject("obs");
        JSONObject unit = findEntityById(obs.getJSONArray("units"), warrior.getActorId());
        assertTrue(unit.optBoolean("hint", false),
                "External payload should flag units that are adjacent to hidden enemy cloaks.");
        assertFalse(hasEntityWithId(obs.getJSONArray("units"), hiddenCloak.getActorId()),
                "External payload should not leak hidden enemy units from the authoritative state.");
        assertFalse(hasEntityWithId(obs.getJSONArray("cities"), bardurCapital.getActorId()),
                "External payload should not leak unexplored enemy cities from the authoritative state.");

        JSONObject lighthouse = obs.getJSONObject("board")
                .getJSONObject("lighthouses").getJSONObject("2");
        assertEquals(4, lighthouse.getInt("x"), "Corner lighthouse metadata should retain the tile x coordinate.");
        assertEquals(0, lighthouse.getInt("y"), "Corner lighthouse metadata should retain the tile y coordinate.");
        JSONArray discoverers = lighthouse.getJSONArray("seen");
        assertEquals(2, discoverers.length(), "Corner lighthouse metadata should include all discoverers.");
        assertEquals(1, discoverers.getInt(0), "Bardur should be listed as a lighthouse discoverer.");
        assertEquals(2, discoverers.getInt(1), "Xin-Xi should be listed as a lighthouse discoverer.");
    }

    private static void testExternalForwardModelStateUsesCompactSchema() {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 1);
        scenario.addUnit(capital, Types.UNIT.WARRIOR, 1, 2);
        GameState gs = scenario.buildState();
        gs.computePlayerActions(gs.getActiveTribe());

        ArrayList<Action> actions = gs.getAllAvailableActions();
        ArrayList<String> actionIds = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            actionIds.add("A" + i);
        }

        ExternalForwardModelSession session = new ExternalForwardModelSession(gs, 0, actions, actionIds);
        JSONObject inspectRequest = new JSONObject();
        inspectRequest.put("command", "inspect");
        JSONObject response = session.handleCommand(inspectRequest);

        assertEquals("forward_model_result", response.getString("type"), "Forward-model inspect should succeed.");
        JSONObject state = response.getJSONObject("state");
        assertTrue(state.has("obs"), "Forward-model state should expose observation under obs.");
        assertTrue(state.has("actions"), "Forward-model state should include actions.");
        assertEquals("A0", state.getJSONArray("actions").getJSONObject(0).getString("id"),
                "Root forward-model state should preserve caller-provided action ids.");
        assertTrue(state.has("active"), "Forward-model state should use compact active key.");
        assertTrue(state.has("terminal"), "Forward-model state should use compact terminal key.");
        assertEquals(Types.MAP_TYPE.CONTINENTS.name(), state.getJSONObject("obs").getString("map"),
                "Forward-model state should expose the current map type.");
        JSONObject tribe = state.getJSONObject("obs").getJSONArray("tribes").getJSONObject(0);
        assertTrue(tribe.has("cities"), "Native parity tribe payload should include city membership.");
        assertTrue(tribe.has("extra"), "Native parity tribe payload should include extra unit ids.");
        assertTrue(tribe.has("conn"), "Native parity tribe payload should include connected city ids.");
        assertTrue(tribe.has("mon"), "Native parity tribe payload should include monument status.");
        JSONObject city = state.getJSONObject("obs").getJSONArray("cities").getJSONObject(0);
        assertTrue(city.has("bound"), "Native parity city payload should include border radius.");
        assertTrue(city.has("units"), "Native parity city payload should include unit membership.");
        assertTrue(city.has("inf"), "Native parity city payload should include infiltration flag.");
        JSONObject unit = state.getJSONObject("obs").getJSONArray("units").getJSONObject(0);
        assertTrue(unit.has("hpx"), "Native parity unit payload should include exact hit points.");
        assertTrue(unit.has("hts"), "Native parity unit payload should include hidden-at-turn-start.");
        assertFalse(state.has("observation"), "Verbose observation key should be removed from forward-model state.");
        assertFalse(state.has("active_player_id"), "Verbose active_player_id key should be removed from forward-model state.");
        assertFalse(state.has("is_terminal"), "Verbose is_terminal key should be removed from forward-model state.");

        JSONObject stepRequest = new JSONObject();
        stepRequest.put("command", "step");
        stepRequest.put("state_id", "root");
        stepRequest.put("i", 0);
        JSONObject stepResponse = session.handleCommand(stepRequest);
        assertEquals("forward_model_result", stepResponse.getString("type"),
                "Forward-model step should accept the compact action index key i.");
        assertEquals(0, stepResponse.getInt("action_index"),
                "Forward-model step should resolve compact action index i to the intended action.");
        JSONObject childState = stepResponse.getJSONObject("state");
        assertTrue(childState.getJSONArray("actions").getJSONObject(0).getString("id").startsWith("s"),
                "Child forward-model states should include stable generated action ids.");
    }

    private static void testExternalGameOverUsesCompactSchema() {
        Scenario scenario = new Scenario(5, Types.GAME_MODE.MIGHT, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        GameState gs = scenario.buildState();
        scenario.tribes[0].setWinner(Types.RESULT.WIN);
        scenario.tribes[1].setWinner(Types.RESULT.LOSS);
        scenario.tribes[0].setScore(5000);
        scenario.tribes[1].setScore(2500);
        gs.computeGameRanking();
        gs.setGameIsOver(true);

        JSONObject payload = ExternalBotPayloadBuilder.buildResult(gs, 0, 1.0);
        assertFalse(payload.has("protocol_version"), "Compact game-over payload should not include a protocol version.");
        assertTrue(payload.has("winner"), "Compact game-over payload should use winner.");
        assertTrue(payload.has("scores"), "Compact game-over payload should use scores.");
        assertTrue(payload.has("rank"), "Compact game-over payload should use rank.");
        assertTrue(payload.has("term"), "Compact game-over payload should use term.");
        assertFalse(payload.has("winner_id"), "Verbose winner_id should be removed.");
        assertFalse(payload.has("final_scores"), "Verbose final_scores should be removed.");
        assertFalse(payload.has("ranking"), "Verbose ranking should be removed.");
        assertFalse(payload.has("ranking_details"), "Verbose ranking_details should be removed.");
        assertFalse(payload.has("final_state"), "Verbose final_state should be removed.");
        assertFalse(payload.has("terminal_reward_contract"), "Verbose terminal reward contract should be removed.");

        JSONObject scoreEntry = findEntityById(payload.getJSONArray("scores"), 0);
        assertTrue(scoreEntry.has("res"), "Final scores should use compact result key res.");
        assertFalse(scoreEntry.has("result"), "Verbose result key should be removed from final scores.");
    }

    private static void testSaveLoadRoundTrip() throws IOException {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS);
        City capital = scenario.addCapital(0, 1, 1);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 1, 2);
        GameState gs = scenario.buildState();

        capital.setPopulation(1);
        warrior.setCurrentHP(7);
        scenario.tribes[0].addStars(8);
        scenario.tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FARMING);

        long seed = System.nanoTime();
        Path saveRoot = Paths.get("save", Long.toString(seed));
        Path saveFile = saveRoot.resolve(gs.getTick() + "_" + gs.getActiveTribeID()).resolve("game.json");

        try {
            Files.createDirectories(saveRoot);
            GameSaver.writeTurnFile(gs, scenario.board, seed);
            assertTrue(Files.exists(saveFile), "Save file should be created");

            GameLoader loader = new GameLoader(saveFile.toString());
            Board loadedBoard = loader.getBoard();
            Tribe loadedTribe = loader.getTribes()[0];
            City loadedCapital = (City) loadedBoard.getActor(capital.getActorId());
            Unit loadedWarrior = (Unit) loadedBoard.getActor(warrior.getActorId());

            assertEquals(3, loader.getSchemaVersion(), "New saves should record the current schema version");
            assertEquals(gs.getTick(), loader.getTick(), "Tick should survive save/load");
            assertEquals(seed, loader.getSeed(), "Seed should survive save/load");
            assertEquals(gs.getGameMode(), loader.getGame_mode(), "Game mode should survive save/load");
            assertEquals(gs.getActiveTribeID(), loader.getActiveTribeID(), "Active tribe should survive save/load");
            assertEquals(Types.TERRAIN.CITY, loadedBoard.getTerrainAt(1, 1), "City terrain should survive save/load");
            assertEquals(scenario.tribes[0].getStars(), loadedTribe.getStars(), "Tribe stars should survive save/load");
            assertEquals(capital.getActorId(), loadedTribe.getCapitalID(), "Capital id should survive save/load");
            assertEquals(capital.getLevel(), loadedCapital.getLevel(), "City level should survive save/load");
            assertEquals(capital.getPopulation(), loadedCapital.getPopulation(), "City population should survive save/load");
            assertEquals(warrior.getCurrentHP(), loadedWarrior.getCurrentHP(), "Unit HP should survive save/load");
            assertVectorEquals(warrior.getPosition(), loadedWarrior.getPosition(), "Unit position should survive save/load");
            assertTrue(loadedTribe.getTechTree().isResearched(Types.TECHNOLOGY.FARMING), "Researched technology should survive save/load");
        } finally {
            deleteRecursivelyQuietly(saveRoot);
        }
    }

    private static void testTurnSensitiveSaveLoadStateSurvivesRoundTrip() throws IOException {
        Scenario scenario = new Scenario(5, Types.TRIBE.IMPERIUS, Types.TRIBE.BARDUR);
        City capital = scenario.addCapital(0, 1, 1);
        scenario.addCapital(1, 3, 3);
        Unit warrior = scenario.addUnit(capital, Types.UNIT.WARRIOR, 1, 2);
        GameState gs = scenario.buildState();

        warrior.setStatus(Types.TURN_STATUS.MOVED);
        warrior.setHidden(true);

        long seed = 6060L;
        Path saveRoot = Paths.get("save", Long.toString(seed));
        Path saveFile = saveRoot.resolve(gs.getTick() + "_" + gs.getActiveTribeID()).resolve("game.json");

        try {
            Files.createDirectories(saveRoot);
            GameSaver.writeTurnFile(gs, scenario.board, seed, true);

            GameLoader loader = new GameLoader(saveFile.toString());
            Tribe loadedTribe = loader.getTribes()[0];
            Unit loadedWarrior = (Unit) loader.getBoard().getActor(warrior.getActorId());

            assertTrue(loader.isPendingTickAdvance(), "Pending tick-advance metadata should survive save/load.");
            assertEquals(Types.TURN_STATUS.MOVED, loadedWarrior.getStatus(), "Unit status should survive save/load.");
            assertTrue(loadedWarrior.isHidden(), "Unit hidden flag should still survive save/load.");
        } finally {
            deleteRecursivelyQuietly(saveRoot);
        }
    }

    private static void deleteRecursively(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }

        try (Stream<Path> walk = Files.walk(root)) {
            walk.sorted((a, b) -> b.compareTo(a)).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException) {
                throw (IOException) e.getCause();
            }
            throw e;
        }
    }

    private static void deleteRecursivelyQuietly(Path root) {
        try {
            deleteRecursively(root);
        } catch (IOException e) {
            System.err.println("[WARN] Could not delete transient save output at " + root + ": " + e.getMessage());
        }
    }

    private static Random extractRandom(GameState state) throws Exception {
        Field field = GameState.class.getDeclaredField("rnd");
        field.setAccessible(true);
        return (Random) field.get(state);
    }

    private static Agent[] extractPlayers(Game game) throws Exception {
        Field field = Game.class.getDeclaredField("players");
        field.setAccessible(true);
        return (Agent[]) field.get(game);
    }

    private static boolean hasResearchAction(GameState state, Types.TECHNOLOGY technology) {
        for (Action action : state.getTribeActions()) {
            if (action instanceof ResearchTech && ((ResearchTech) action).getTech() == technology) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasTribeActionType(GameState state, Types.ACTION actionType) {
        for (Action action : state.getTribeActions()) {
            if (action.getActionType() == actionType) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasUnitActionType(GameState state, Types.ACTION actionType) {
        for (Integer unitId : state.getUnitActions().keySet()) {
            for (Action action : state.getUnitActions(unitId)) {
                if (action.getActionType() == actionType) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean hasSpawnAction(GameState state, int cityId, Types.UNIT unitType) {
        if (state.getCityActions(cityId) == null) {
            return false;
        }

        for (Action action : state.getCityActions(cityId)) {
            if (action instanceof Spawn && ((Spawn) action).getUnitType() == unitType) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasMoveTo(java.util.List<Action> actions, int x, int y) {
        for (Action action : actions) {
            if (action instanceof Move) {
                Vector2d destination = ((Move) action).getDestination();
                if (destination.x == x && destination.y == y) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int countExtraUnitsOfType(GameState state, int tribeId, Types.UNIT unitType) {
        int count = 0;
        for (Integer unitId : state.getTribe(tribeId).getExtraUnits()) {
            Unit unit = (Unit) state.getActor(unitId);
            if (unit != null && unit.getType() == unitType) {
                count++;
            }
        }
        return count;
    }

    private static int countResearchedTechs(TechnologyTree techTree) {
        int researched = 0;
        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            if (techTree.isResearched(technology)) {
                researched++;
            }
        }
        return researched;
    }

    private static JSONObject findEntityById(JSONArray entities, int id) {
        for (int i = 0; i < entities.length(); i++) {
            JSONObject entity = entities.getJSONObject(i);
            if (entity.getInt("id") == id) {
                return entity;
            }
        }
        throw new AssertionError("Could not find entity with id " + id + ".");
    }

    private static boolean hasEntityWithId(JSONArray entities, int id) {
        for (int i = 0; i < entities.length(); i++) {
            JSONObject entity = entities.getJSONObject(i);
            if (entity.getInt("id") == id) {
                return true;
            }
        }
        return false;
    }

    private static void addBuilding(GameState gs, City city, Types.BUILDING buildingType, int x, int y) {
        gs.getBoard().setBuildingAt(x, y, buildingType);
        if (buildingType.isTemple()) {
            city.addBuilding(gs, new Temple(x, y, buildingType, city.getActorId()));
        } else {
            city.addBuilding(gs, new Building(x, y, buildingType, city.getActorId()));
        }
        if (buildingType == Types.BUILDING.PORT) {
            gs.getBoard().buildPort(x, y);
        }
    }

    private static void buildRoad(GameState gs, int tribeId, int x, int y) {
        BuildRoad buildRoad = new BuildRoad(tribeId);
        buildRoad.setPosition(new Vector2d(x, y));
        assertTrue(buildRoad.isFeasible(gs), "Expected road/bridge build to be feasible at " + x + "," + y + ".");
        assertTrue(new BuildRoadCommand().execute(buildRoad, gs),
                "Road/bridge build should execute at " + x + "," + y + ".");
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void assertFalse(boolean condition, String message) {
        assertTrue(!condition, message);
    }

    private static void assertEquals(int expected, int actual, String message) {
        if (expected != actual) {
            throw new AssertionError(message + " (expected " + expected + ", got " + actual + ")");
        }
    }

    private static void assertEquals(long expected, long actual, String message) {
        if (expected != actual) {
            throw new AssertionError(message + " (expected " + expected + ", got " + actual + ")");
        }
    }

    private static void assertEquals(double expected, double actual, double tolerance, String message) {
        if (Math.abs(expected - actual) > tolerance) {
            throw new AssertionError(message + " (expected " + expected + ", got " + actual + ")");
        }
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (expected == null ? actual != null : !expected.equals(actual)) {
            throw new AssertionError(message + " (expected " + expected + ", got " + actual + ")");
        }
    }

    private static void assertSame(Object expected, Object actual, String message) {
        if (expected != actual) {
            throw new AssertionError(message + " (expected reference " + expected + ", got " + actual + ")");
        }
    }

    private static void assertVectorEquals(Vector2d expected, Vector2d actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " (expected " + expected + ", got " + actual + ")");
        }
    }

    @FunctionalInterface
    private interface CheckedRunnable {
        void run() throws Exception;
    }

    private static final class HarnessTest {
        private final String name;
        private final CheckedRunnable check;

        private HarnessTest(String name, CheckedRunnable check) {
            this.name = name;
            this.check = check;
        }
    }

    private static final class HarnessSuite {
        private final String name;
        private final HarnessTest[] tests;

        private HarnessSuite(String name, HarnessTest[] tests) {
            this.name = name;
            this.tests = tests;
        }
    }

    private static final class GeneratedLevel {
        private final int size;
        private final Types.TERRAIN[][] terrain;
        private final Types.RESOURCE[][] resource;
        private final String[][] raw;

        private GeneratedLevel(int size, Types.TERRAIN[][] terrain, Types.RESOURCE[][] resource, String[] lines) {
            this.size = size;
            this.terrain = terrain;
            this.resource = resource;
            this.raw = new String[size][size];
            for (int row = 0; row < size; row++) {
                this.raw[row] = lines[row].split(",");
            }
        }
    }

    private static final class FixedRandom extends Random {
        private final int[] values;
        private int index;

        private FixedRandom(int... values) {
            super(0L);
            this.values = values.length == 0 ? new int[]{0} : values.clone();
        }

        @Override
        public int nextInt(int bound) {
            int value = values[Math.min(index, values.length - 1)];
            index++;
            return Math.floorMod(value, bound);
        }
    }

    private static final class Scenario {
        private final Random random = new Random(SCENARIO_SEED);
        private final Types.GAME_MODE gameMode;
        private final Board board = new Board();
        private final Tribe[] tribes;

        private Scenario(int size, Types.TRIBE... tribeTypes) {
            this(size, Types.GAME_MODE.SCORE, tribeTypes);
        }

        private Scenario(int size, Types.GAME_MODE gameMode, Types.TRIBE... tribeTypes) {
            this.gameMode = gameMode;
            this.tribes = new Tribe[tribeTypes.length];
            for (int i = 0; i < tribeTypes.length; i++) {
                this.tribes[i] = new Tribe(tribeTypes[i]);
            }

            board.init(size, tribes);
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    board.setTerrainAt(x, y, Types.TERRAIN.PLAIN);
                }
            }
        }

        private City addCapital(int tribeId, int x, int y) {
            return addCity(tribeId, x, y, true);
        }

        private City addCity(int tribeId, int x, int y) {
            return addCity(tribeId, x, y, false);
        }

        private City addCity(int tribeId, int x, int y, boolean capital) {
            City city = new City(x, y, tribeId);
            city.setCapital(capital);
            board.setTerrainAt(x, y, Types.TERRAIN.CITY);
            board.addCityToTribe(city, random);
            board.assignCityTiles(city, city.getBound());
            tribes[tribeId].addScore(TribesConfig.CITY_CENTRE_POINTS);
            return city;
        }

        private Unit addUnit(City homeCity, Types.UNIT unitType, int x, int y) {
            Unit unit = Types.UNIT.createUnit(new Vector2d(x, y), 0, false, homeCity.getActorId(), homeCity.getTribeId(), unitType);
            board.addUnit(homeCity, unit);
            tribes[homeCity.getTribeId()].addScore(unitType.getPoints());
            return unit;
        }

        private GameState buildState() {
            return buildState(new Random(SCENARIO_SEED), 0, GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
        }

        private GameState buildState(int tick) {
            return buildState(new Random(SCENARIO_SEED), tick, GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
        }

        private GameState buildState(Random stateRandom) {
            return buildState(stateRandom, 0, GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
        }

        private GameState buildState(int tick, int gloryTargetScore) {
            return buildState(new Random(SCENARIO_SEED), tick, gloryTargetScore);
        }

        private GameState buildState(Random stateRandom, int tick, int gloryTargetScore) {
            board.setActiveTribeID(0);
            GameState state = new GameState(stateRandom, gameMode, gloryTargetScore, tribes, board, tick);
            state.setMapType(Types.MAP_TYPE.CONTINENTS);
            board.revealExplorationFromCurrentAssets(stateRandom);
            return state;
        }
    }

    private static final class SeatTrackingAgent extends Agent {
        private final String label;

        private SeatTrackingAgent(long seed, String label) {
            super(seed);
            this.label = label;
        }

        @Override
        public Action act(GameState gs, ElapsedCpuTimer ect) {
            return null;
        }

        @Override
        public Agent copy() {
            return new SeatTrackingAgent(getSeed(), label);
        }
    }
}

