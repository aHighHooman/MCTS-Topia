package core.levelgen;

import static core.Types.RESOURCE.*;
import static core.Types.TERRAIN.*;
import static core.Types.TRIBE.*;
//import static core.Types.RESOURCE.*;
import core.Types;
import org.json.JSONObject;
import utils.file.IO;

import java.io.FileWriter;
import java.util.*;

/**
 * This is a Java port of the level generator created for the game Polytopia adapted to our format.
 * Original source: https://github.com/QuasiStellar/Polytopia-Map-Generator.
 */

public class LevelGenerator {

    //Level parameters, can be changed using init().
    private int mapSize;
    private int smoothing;
    private int relief;
    private double initialLand;
    private double landCoefficient;
    private String[] level;
    private Types.TRIBE[] biomeOwner;
    private Types.TRIBE[] tribes;
    private Types.MAP_TYPE mapType;
    private GenerationProfile profile;
    private double BORDER_EXPANSION = 1/3.0;
    private long seed;
    private Random rnd;

    //JSON that contains all the probability values for all the tribes.
    private static JSONObject cachedTerrainProbabilityData;
    private JSONObject data;

    private boolean LEVELGEN_VERBOSE = false;

    /**
     * Constructor of the generator
     */
    public LevelGenerator(long seed) {

        this.seed = seed;
        this.rnd = new Random(mixSeed(seed));

        //Initialize with default values.
        init(Types.MAP_SIZE.TINY.getSideLength(), Types.MAP_TYPE.CONTINENTS, new Types.TRIBE[]{XIN_XI, OUMAJI});

        // Read the JSON that contains all the probability values for all the tribes.
        this.data = terrainProbabilityData();
    }

    private static synchronized JSONObject terrainProbabilityData() {
        if (cachedTerrainProbabilityData == null) {
            cachedTerrainProbabilityData = new IO().readJSON("terrainProbs.json");
        }
        return cachedTerrainProbabilityData;
    }

    /**
     * SplitMix64-style avalanche mixer so nearby or single-bit-different seeds
     * produce decorrelated RNG streams for map generation.
     */
    private long mixSeed(long value) {
        long mixed = value;
        mixed ^= (mixed >>> 30);
        mixed *= 0xbf58476d1ce4e5b9L;
        mixed ^= (mixed >>> 27);
        mixed *= 0x94d049bb133111ebL;
        mixed ^= (mixed >>> 31);
        return mixed;
    }

    /**
     * Initializes the map generation parameters.
     */
    public void init(int mapSize, Types.MAP_TYPE mapType, Types.TRIBE[] tribes) {
        Types.MAP_SIZE supportedSize = Types.MAP_SIZE.fromSideLength(mapSize);
        if (supportedSize == null) {
            throw new IllegalArgumentException("Unsupported map size side length: " + mapSize);
        }
        if (mapType == null) {
            throw new IllegalArgumentException("Map type cannot be null.");
        }
        if (tribes == null || tribes.length == 0) {
            throw new IllegalArgumentException("At least one tribe is required for level generation.");
        }

        this.mapSize = mapSize;
        this.level = new String[mapSize*mapSize];
        this.biomeOwner = new Types.TRIBE[mapSize*mapSize];
        this.tribes = tribes;
        this.mapType = mapType;
        configureMapType(mapType);

        //Initialize the level with deep water.
        for(int i = 0; i < mapSize*mapSize; i++){ level[i] = "d: "; };
    }

    /**
     * Generates the level.
     */
    public void generate() {

        if (LEVELGEN_VERBOSE) System.out.println("Generating level with seed: " + this.seed);
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            level[cell] = DEEP_WATER.getMapChar() + ": ";
        }
        this.profile = GenerationProfile.forMapType(mapType, mapSize);

        if (profile.capitalsFromVillages) {
            generateVillageCapitalMap();
        } else {
            generateQuadrantCapitalMap();
        }
    }

    private void generateVillageCapitalMap() {
        if (mapType == Types.MAP_TYPE.CONTINENTS) {
            generateContinentsLand();
        } else if (mapType == Types.MAP_TYPE.PANGEA) {
            generatePangeaLand();
        } else {
            generateLandFromWetness();
        }
        if (profile.edgeLandBridge) {
            forceEdgeLandBridge();
        }
        normalizeDeepWaterCoastlines();

        ArrayList<Integer> villageCells = new ArrayList<>();
        villageCells.addAll(placePostTerrainVillages());
        if (mapType == Types.MAP_TYPE.CONTINENTS) {
            ensureContinentsHaveVillageNearEachCapitalCandidate(villageCells);
        }
        ArrayList<Integer> capitalCells = selectCapitalVillages(villageCells);
        convertVillagesToCapitals(capitalCells);
        villageCells.addAll(placeTinyIslandVillages());

        Types.TRIBE[] tileOwner = assignTileOwners(capitalCells);
        this.biomeOwner = tileOwner;
        generateTerrain(tileOwner);
        normalizeDeepWaterCoastlines();

        generateResources(tileOwner);
        placeCornerLighthouses();
        normalizeDeepWaterCoastlines();
        adjustStartingResources(capitalCells);
        normalizeDeepWaterCoastlines();
        placeRuins();
        if (mapType != Types.MAP_TYPE.DRYLANDS) {
            placeStarfish();
        }
    }

    private void generateQuadrantCapitalMap() {
        ArrayList<Integer> capitalCells = selectQuadrantCapitalsFromDomains();
        writeCapitals(capitalCells);

        ArrayList<Integer> villageCells = new ArrayList<>();
        villageCells.addAll(placeSuburbVillages(capitalCells));
        villageCells.addAll(placePreTerrainVillages(capitalCells, villageCells));

        generateLandFromWetness();
        forceSettlementAdjacentLand(capitalCells, villageCells);
        if (profile.edgeLandBridge) {
            forceEdgeLandBridge();
        }
        enforceQuadrantWaterBand();
        normalizeDeepWaterCoastlines();

        Types.TRIBE[] tileOwner = assignTileOwners(capitalCells);
        this.biomeOwner = tileOwner;
        generateTerrain(tileOwner);
        normalizeDeepWaterCoastlines();
        villageCells.addAll(placePostTerrainVillages());
        if (mapType == Types.MAP_TYPE.LAKES) {
            ensureLakesCapitalVillageConnections(capitalCells, villageCells);
        }
        enforceQuadrantWaterBand();

        generateResources(tileOwner);
        placeCornerLighthouses();
        normalizeDeepWaterCoastlines();
        adjustStartingResources(capitalCells);
        normalizeDeepWaterCoastlines();
        enforceQuadrantWaterBand();
        if (mapType != Types.MAP_TYPE.DRYLANDS) {
            forceSettlementAdjacentLand(capitalCells, currentVillageCells());
            normalizeDeepWaterCoastlines();
            enforceQuadrantWaterBand();
        }
        normalizeDeepWaterCoastlines();
        placeRuins();
        if (mapType != Types.MAP_TYPE.DRYLANDS) {
            placeStarfish();
        }
    }

    private void generateLandFromWetness() {
        if (LEVELGEN_VERBOSE) System.out.println("Generate map-type land/water.");
        int targetLand = (int) Math.round(mapSize * mapSize * profile.landRatio);
        int placed = 0;
        while (placed < targetLand) {
            int index = randomInt(0, mapSize * mapSize);
            if (getTerrain(index) == DEEP_WATER.getMapChar()) {
                placed++;
                writeTile(index, "" + PLAIN.getMapChar(), null);
            }
        }

        for (int pass = 0; pass < profile.smoothing; pass++) {
            ArrayList<Integer> land = new ArrayList<>();
            ArrayList<Integer> water = new ArrayList<>();
            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                int waterCount = 0;
                int total = 0;
                for (int n : disk(cell, 1)) {
                    if (getTerrain(n) == DEEP_WATER.getMapChar()) {
                        waterCount++;
                    }
                    total++;
                }
                if (waterCount / (double) total <= profile.landCoefficient) {
                    land.add(cell);
                } else {
                    water.add(cell);
                }
            }
            for (int cell : land) {
                if (!isSettlement(cell)) {
                    writeTile(cell, "" + PLAIN.getMapChar(), null);
                }
            }
            for (int cell : water) {
                if (!isSettlement(cell)) {
                    writeTile(cell, "" + DEEP_WATER.getMapChar(), null);
                }
            }
        }

        if (mapType == Types.MAP_TYPE.WATER_WORLD) {
            int islandCount = Math.max(tribes.length * 2, mapSize / 2);
            for (int i = 0; i < islandCount; i++) {
                int center = randomInteriorCell(2);
                if (!isSettlement(center)) {
                    writeTile(center, "" + PLAIN.getMapChar(), null);
                }
                if (rnd.nextBoolean()) {
                    ArrayList<Integer> neighbours = crossNeighbors(center);
                    int neighbour = neighbours.get(randomInt(0, neighbours.size()));
                    if (!isSettlement(neighbour)) {
                        writeTile(neighbour, "" + PLAIN.getMapChar(), null);
                    }
                }
            }
        }
    }

    private void generatePangeaLand() {
        if (LEVELGEN_VERBOSE) System.out.println("Generate Pangea main landmass.");
        int targetLand = (int) Math.round(mapSize * mapSize * profile.landRatio);
        int center = (mapSize / 2) * mapSize + (mapSize / 2);
        ArrayList<Integer> frontier = new ArrayList<>();
        HashSet<Integer> land = new HashSet<>();
        frontier.add(center);
        land.add(center);
        writeTile(center, "" + PLAIN.getMapChar(), null);

        while (land.size() < targetLand && !frontier.isEmpty()) {
            int source = frontier.get(randomInt(0, frontier.size()));
            ArrayList<Integer> neighbours = circle(source, 1);
            Collections.shuffle(neighbours, rnd);
            boolean expanded = false;
            for (int neighbour : neighbours) {
                if (!land.contains(neighbour)) {
                    land.add(neighbour);
                    frontier.add(neighbour);
                    writeTile(neighbour, "" + PLAIN.getMapChar(), null);
                    expanded = true;
                    break;
                }
            }
            if (!expanded) {
                frontier.remove(Integer.valueOf(source));
            }
        }
    }

    private void generateContinentsLand() {
        if (LEVELGEN_VERBOSE) System.out.println("Generate separated Continents landmasses.");
        int targetLand = (int) Math.round(mapSize * mapSize * profile.landRatio);
        int continentCount = Math.max(2, Math.min(tribes.length + 1, mapSize / 4));
        int[] component = new int[mapSize * mapSize];
        Arrays.fill(component, -1);

        ArrayList<ArrayList<Integer>> frontiers = new ArrayList<>();
        int placedLand = 0;
        for (int continent = 0; continent < continentCount; continent++) {
            int seed = findContinentSeed(component);
            if (seed < 0) {
                break;
            }
            component[seed] = continent;
            writeTile(seed, "" + PLAIN.getMapChar(), null);
            ArrayList<Integer> frontier = new ArrayList<>();
            frontier.add(seed);
            frontiers.add(frontier);
            placedLand++;
        }

        while (placedLand < targetLand) {
            ArrayList<Integer> expandable = new ArrayList<>();
            for (int continent = 0; continent < frontiers.size(); continent++) {
                if (hasContinentExpansionCell(frontiers.get(continent), component, continent)) {
                    expandable.add(continent);
                }
            }
            if (expandable.isEmpty()) {
                break;
            }
            int continent = expandable.get(randomInt(0, expandable.size()));
            Integer next = pickContinentExpansionCell(frontiers.get(continent), component, continent);
            if (next == null) {
                break;
            }
            component[next] = continent;
            writeTile(next, "" + PLAIN.getMapChar(), null);
            frontiers.get(continent).add(next);
            placedLand++;
        }
    }

    private void writeCapitals(ArrayList<Integer> capitalCells) {
        for (int i = 0; i < capitalCells.size(); i++) {
            level[capitalCells.get(i)] = CITY.getMapChar() + ":" + tribes[i].getKey() + ":" + i;
        }
    }

    private void convertVillagesToCapitals(ArrayList<Integer> capitalCells) {
        for (int i = 0; i < capitalCells.size(); i++) {
            int capital = capitalCells.get(i);
            level[capital] = CITY.getMapChar() + ":" + tribes[i].getKey() + ":" + i;
        }
    }

    private ArrayList<Integer> placeSuburbVillages(ArrayList<Integer> capitalCells) {
        ArrayList<Integer> villages = new ArrayList<>();
        if (!profile.suburbs) {
            return villages;
        }
        for (int capital : capitalCells) {
            int placed = 0;
            ArrayList<Integer> candidates = new ArrayList<>(disk(capital, Math.max(2, mapSize / 4)));
            Collections.shuffle(candidates, rnd);
            for (int cell : candidates) {
                if (placed >= 2) {
                    break;
                }
                boolean canPlace = mapType.usesQuadrantCapitals()
                        ? canPlacePreTerrainVillage(cell, villages, 1, 2)
                        : canPlaceVillage(cell, villages, 1, 2);
                if (canPlace) {
                    writeTile(cell, "" + VILLAGE.getMapChar(), null);
                    villages.add(cell);
                    placed++;
                }
            }
        }
        return villages;
    }

    private ArrayList<Integer> placePreTerrainVillages(ArrayList<Integer> capitalCells, ArrayList<Integer> existingVillages) {
        ArrayList<Integer> villages = new ArrayList<>();
        if (profile.preTerrainVillageCoefficient <= 0.0) {
            return villages;
        }
        int domain = mapSize / 3;
        int desired = (int) Math.round((domain * domain - capitalCells.size() - existingVillages.size())
                * profile.preTerrainVillageCoefficient);
        desired = Math.max(0, desired);
        ArrayList<Integer> placed = new ArrayList<>(existingVillages);
        while (villages.size() < desired) {
            ArrayList<Integer> candidates = mapType.usesQuadrantCapitals()
                    ? preTerrainVillageCandidates(placed, 1, 2)
                    : villageCandidates(placed, 1, 2);
            if (candidates.isEmpty()) {
                break;
            }
            int village = candidates.get(randomInt(0, candidates.size()));
            writeTile(village, "" + VILLAGE.getMapChar(), null);
            villages.add(village);
            placed.add(village);
        }
        return villages;
    }

    private ArrayList<Integer> placePostTerrainVillages() {
        ArrayList<Integer> villages = currentVillageCells();
        ArrayList<Integer> added = new ArrayList<>();
        while (true) {
            ArrayList<Integer> candidates = villageCandidates(villages, profile.postTerrainVillageEdgeBuffer, 2);
            if (candidates.isEmpty()) {
                break;
            }
            int village = candidates.get(randomInt(0, candidates.size()));
            writeTile(village, "" + VILLAGE.getMapChar(), null);
            villages.add(village);
            added.add(village);
        }
        return added;
    }

    private ArrayList<Integer> placeTinyIslandVillages() {
        ArrayList<Integer> islands = new ArrayList<>();
        if (profile.tinyIslandVillages <= 0) {
            return islands;
        }
        for (int i = 0; i < profile.tinyIslandVillages; i++) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                if (getTerrain(cell) == DEEP_WATER.getMapChar()
                        && getResource(cell).isEmpty()
                        && !nearLand(cell, 1)
                        && !nearAny(cell, currentVillageCells(), 2)
                        && !nearMapEdge(cell, 1)) {
                    candidates.add(cell);
                }
            }
            if (candidates.isEmpty()) {
                break;
            }
            int center = candidates.get(randomInt(0, candidates.size()));
            writeTile(center, "" + VILLAGE.getMapChar(), null);
            islands.add(center);
            normalizeDeepWaterCoastlines();
        }
        return islands;
    }

    private ArrayList<Integer> selectCapitalVillages(ArrayList<Integer> villageCells) {
        ArrayList<Integer> capitals = new ArrayList<>();
        ArrayList<Integer> available = new ArrayList<>();
        for (int village : villageCells) {
            if (!nearMapEdge(village, 2)) {
                available.add(village);
            }
        }
        if (available.size() < tribes.length) {
            while (available.size() < tribes.length) {
                ArrayList<Integer> candidates = villageCandidates(available, 2, 2);
                if (candidates.isEmpty()) {
                    break;
                }
                int village = candidates.get(randomInt(0, candidates.size()));
                writeTile(village, "" + VILLAGE.getMapChar(), null);
                available.add(village);
            }
        }
        if (mapType == Types.MAP_TYPE.CONTINENTS) {
            ensureContinentsHaveVillageNearEachCapitalCandidate(available);
        }
        HashSet<Integer> usedLandComponents = new HashSet<>();
        for (int i = 0; i < tribes.length && !available.isEmpty(); i++) {
            ArrayList<Integer> componentCandidates = new ArrayList<>();
            if (mapType == Types.MAP_TYPE.CONTINENTS) {
                for (int candidate : available) {
                    int component = landComponentKey(candidate);
                    if (!usedLandComponents.contains(component)
                            && countVillagesOnLandComponent(available, component) > 1) {
                        componentCandidates.add(candidate);
                    }
                }
                if (componentCandidates.isEmpty()) {
                    for (int candidate : available) {
                        int component = landComponentKey(candidate);
                        if (!usedLandComponents.contains(component)) {
                            componentCandidates.add(candidate);
                        }
                    }
                }
            }
            ArrayList<Integer> candidates = componentCandidates.isEmpty() ? available : componentCandidates;
            int chosen = pickBestVillageCapital(candidates, capitals, profile.coastalCapitals);
            capitals.add(chosen);
            usedLandComponents.add(landComponentKey(chosen));
            available.remove(Integer.valueOf(chosen));
        }
        return capitals;
    }

    private void ensureContinentsHaveVillageNearEachCapitalCandidate(ArrayList<Integer> villageCells) {
        HashSet<Integer> componentKeys = new HashSet<>();
        for (int village : villageCells) {
            componentKeys.add(landComponentKey(village));
        }

        for (Integer component : new ArrayList<>(componentKeys)) {
            ArrayList<Integer> componentVillages = new ArrayList<>();
            for (int village : villageCells) {
                if (landComponentKey(village) == component) {
                    componentVillages.add(village);
                }
            }
            addVillageToComponentIfPossible(villageCells, component, 2, profile.postTerrainVillageEdgeBuffer);
        }

        while (countComponentsWithMultipleVillages(villageCells) < tribes.length) {
            Integer component = findComponentNeedingVillage(villageCells, new HashSet<Integer>());
            if (component == null || !expandComponentUntilVillageFits(villageCells, component)) {
                break;
            }
        }
    }

    private boolean addVillageToComponentIfPossible(ArrayList<Integer> villageCells, int component, int spacing, int edgeBuffer) {
        if (countVillagesOnLandComponent(villageCells, component) >= 2) {
            return true;
        }
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (getTerrain(cell) != PLAIN.getMapChar() && getTerrain(cell) != FOREST.getMapChar()) {
                continue;
            }
            if (!getResource(cell).isEmpty() || nearMapEdge(cell, edgeBuffer)) {
                continue;
            }
            if (landComponentKey(cell) != component || nearAny(cell, villageCells, spacing)) {
                continue;
            }
            candidates.add(cell);
        }
        if (candidates.isEmpty()) {
            return false;
        }
        int village = candidates.get(randomInt(0, candidates.size()));
        writeTile(village, "" + VILLAGE.getMapChar(), null);
        villageCells.add(village);
        return true;
    }

    private int countComponentsWithMultipleVillages(ArrayList<Integer> villageCells) {
        HashSet<Integer> counted = new HashSet<>();
        for (int village : villageCells) {
            int component = landComponentKey(village);
            if (countVillagesOnLandComponent(villageCells, component) > 1) {
                counted.add(component);
            }
        }
        return counted.size();
    }

    private Integer findComponentNeedingVillage(ArrayList<Integer> villageCells, HashSet<Integer> skippedComponents) {
        HashSet<Integer> seen = new HashSet<>();
        for (int village : villageCells) {
            int component = landComponentKey(village);
            if (!seen.contains(component)
                    && !skippedComponents.contains(component)
                    && countVillagesOnLandComponent(villageCells, component) < 2) {
                return component;
            }
            seen.add(component);
        }
        return null;
    }

    private boolean expandComponentUntilVillageFits(ArrayList<Integer> villageCells, int component) {
        HashSet<Integer> skippedComponents = new HashSet<>();
        for (int attempt = 0; attempt < mapSize * mapSize; attempt++) {
            if (addVillageToComponentIfPossible(villageCells, component, 2, profile.postTerrainVillageEdgeBuffer)) {
                return true;
            }
            if (!expandLandComponent(component)) {
                skippedComponents.add(component);
                Integer nextComponent = findComponentNeedingVillage(villageCells, skippedComponents);
                if (nextComponent == null) {
                    return false;
                }
                component = nextComponent;
            }
            normalizeDeepWaterCoastlines();
        }
        return false;
    }

    private boolean expandLandComponent(int component) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (landComponentKey(cell) != component) {
                continue;
            }
            for (int neighbour : crossNeighbors(cell)) {
                if (isWaterTerrain(getTerrain(neighbour)) && !touchesOtherLandComponent(neighbour, component)) {
                    candidates.add(neighbour);
                }
            }
        }
        if (candidates.isEmpty()) {
            return false;
        }
        int newLand = candidates.get(randomInt(0, candidates.size()));
        writeTile(newLand, "" + PLAIN.getMapChar(), null);
        return true;
    }

    private boolean touchesOtherLandComponent(int cell, int allowedComponent) {
        for (int nearby : disk(cell, 1)) {
            if (!isWaterTerrain(getTerrain(nearby)) && landComponentKey(nearby) != allowedComponent) {
                return true;
            }
        }
        return false;
    }

    private int countVillagesOnLandComponent(ArrayList<Integer> villages, int component) {
        int count = 0;
        for (int village : villages) {
            if (landComponentKey(village) == component) {
                count++;
            }
        }
        return count;
    }

    private void generateTerrain(Types.TRIBE[] tileOwner) {
        if (LEVELGEN_VERBOSE) System.out.println("Generate terrain by tribe biome.");
        HashMap<Types.TRIBE, ArrayList<Integer>> plainCellsByOwner = new HashMap<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (getTerrain(cell) != PLAIN.getMapChar()) {
                continue;
            }
            Types.TRIBE owner = tileOwner[cell] == null ? LUXIDOOR : tileOwner[cell];
            if (!plainCellsByOwner.containsKey(owner)) {
                plainCellsByOwner.put(owner, new ArrayList<Integer>());
            }
            plainCellsByOwner.get(owner).add(cell);
        }

        for (Map.Entry<Types.TRIBE, ArrayList<Integer>> entry : plainCellsByOwner.entrySet()) {
            Types.TRIBE owner = entry.getKey();
            ArrayList<Integer> cells = entry.getValue();
            Collections.shuffle(cells, rnd);

            TerrainRates rates = terrainRates(owner);
            int mountainCount = (int) Math.round(cells.size() * rates.mountain);
            int forestCount = (int) Math.round(cells.size() * rates.forest);
            if (mountainCount + forestCount > cells.size()) {
                forestCount = Math.max(0, cells.size() - mountainCount);
            }

            int index = 0;
            for (; index < mountainCount; index++) {
                writeTile(cells.get(index), "" + MOUNTAIN.getMapChar(), null);
            }
            for (; index < mountainCount + forestCount; index++) {
                writeTile(cells.get(index), "" + FOREST.getMapChar(), null);
            }
        }
    }

    private TerrainRates terrainRates(Types.TRIBE owner) {
        double mountain = clamp(0.14 * getTribeProb("MOUNTAIN", owner), 0.0, 0.85);
        double nonMountain = 1.0 - mountain;
        double baseNonMountain = 1.0 - 0.14;
        double forest = nonMountain * (0.38 / baseNonMountain);
        forest = clamp(forest * getTribeProb("FOREST", owner), 0.0, nonMountain);
        return new TerrainRates(mountain, forest);
    }

    private Types.TRIBE[] assignTileOwners(ArrayList<Integer> capitalCells) {
        Types.TRIBE[] owners = new Types.TRIBE[mapSize * mapSize];
        if (capitalCells.isEmpty()) {
            Arrays.fill(owners, LUXIDOOR);
            return owners;
        }

        int seats = Math.min(capitalCells.size(), tribes.length);
        int[] targetCounts = balancedBiomeTargets(seats);
        int[] counts = new int[seats];
        ArrayList<HashSet<Integer>> frontierBySeat = new ArrayList<>();
        for (int seat = 0; seat < seats; seat++) {
            frontierBySeat.add(new HashSet<Integer>());
        }

        ArrayList<Integer> seatOrder = new ArrayList<>();
        for (int seat = 0; seat < seats; seat++) {
            seatOrder.add(seat);
        }
        Collections.shuffle(seatOrder, rnd);
        for (int seat : seatOrder) {
            int capital = capitalCells.get(seat);
            if (owners[capital] == null) {
                owners[capital] = tribes[seat];
                counts[seat]++;
                addUnownedNeighboursToFrontier(capital, owners, frontierBySeat.get(seat));
            }
        }

        int assigned = 0;
        for (int count : counts) {
            assigned += count;
        }
        while (assigned < mapSize * mapSize) {
            ArrayList<Integer> expandableSeats = new ArrayList<>();
            for (int seat = 0; seat < seats; seat++) {
                pruneOwnedFrontier(frontierBySeat.get(seat), owners);
                if (counts[seat] < targetCounts[seat] && !frontierBySeat.get(seat).isEmpty()) {
                    expandableSeats.add(seat);
                }
            }

            if (expandableSeats.isEmpty()) {
                Integer fallbackCell = firstUnownedCell(owners);
                if (fallbackCell == null) {
                    break;
                }
                int seat = pickLeastFilledNearestSeat(fallbackCell, capitalCells, counts, targetCounts, seats);
                owners[fallbackCell] = tribes[seat];
                counts[seat]++;
                assigned++;
                addUnownedNeighboursToFrontier(fallbackCell, owners, frontierBySeat.get(seat));
                continue;
            }

            int seat = expandableSeats.get(randomInt(0, expandableSeats.size()));
            ArrayList<Integer> frontier = new ArrayList<>(frontierBySeat.get(seat));
            int cell = frontier.get(randomInt(0, frontier.size()));
            frontierBySeat.get(seat).remove(cell);
            if (owners[cell] != null) {
                continue;
            }
            owners[cell] = tribes[seat];
            counts[seat]++;
            assigned++;
            addUnownedNeighboursToFrontier(cell, owners, frontierBySeat.get(seat));
        }

        return owners;
    }

    private int[] balancedBiomeTargets(int seats) {
        int total = mapSize * mapSize;
        int[] targets = new int[seats];
        int base = total / seats;
        int remainder = total % seats;
        ArrayList<Integer> order = new ArrayList<>();
        for (int seat = 0; seat < seats; seat++) {
            order.add(seat);
            targets[seat] = base;
        }
        Collections.shuffle(order, rnd);
        for (int i = 0; i < remainder; i++) {
            targets[order.get(i)]++;
        }
        return targets;
    }

    private void addUnownedNeighboursToFrontier(int cell, Types.TRIBE[] owners, HashSet<Integer> frontier) {
        for (int neighbour : crossNeighbors(cell)) {
            if (owners[neighbour] == null) {
                frontier.add(neighbour);
            }
        }
    }

    private void pruneOwnedFrontier(HashSet<Integer> frontier, Types.TRIBE[] owners) {
        Iterator<Integer> iterator = frontier.iterator();
        while (iterator.hasNext()) {
            if (owners[iterator.next()] != null) {
                iterator.remove();
            }
        }
    }

    private Integer firstUnownedCell(Types.TRIBE[] owners) {
        int start = randomInt(0, owners.length);
        for (int offset = 0; offset < owners.length; offset++) {
            int cell = (start + offset) % owners.length;
            if (owners[cell] == null) {
                return cell;
            }
        }
        return null;
    }

    private int pickLeastFilledNearestSeat(int cell, ArrayList<Integer> capitalCells, int[] counts, int[] targets, int seats) {
        double bestFill = Double.MAX_VALUE;
        int bestDistance = Integer.MAX_VALUE;
        ArrayList<Integer> best = new ArrayList<>();
        for (int seat = 0; seat < seats; seat++) {
            double fill = counts[seat] / (double) Math.max(1, targets[seat]);
            int distance = distance(cell, capitalCells.get(seat), mapSize);
            if (fill < bestFill || (fill == bestFill && distance < bestDistance)) {
                bestFill = fill;
                bestDistance = distance;
                best.clear();
                best.add(seat);
            } else if (fill == bestFill && distance == bestDistance) {
                best.add(seat);
            }
        }
        return best.get(randomInt(0, best.size()));
    }

    public Types.TRIBE getBiomeOwner(int index) {
        if (biomeOwner == null || index < 0 || index >= biomeOwner.length) {
            return null;
        }
        return biomeOwner[index];
    }

    private String tileForOutput(int index) {
        Types.TRIBE owner = getBiomeOwner(index);
        if (owner == null) {
            return level[index];
        }
        String[] pieces = level[index].split(":", -1);
        String resource = pieces.length > 1 && !pieces[1].equals(" ") ? pieces[1] : "";
        if (getTerrain(index) == CITY.getMapChar() && pieces.length >= 3) {
            return pieces[0] + ":" + pieces[1] + ":" + pieces[2] + ":" + owner.getKey();
        }
        return pieces[0] + ":" + resource + ":" + owner.getKey();
    }

    private void generateResources(Types.TRIBE[] tileOwner) {
        if (LEVELGEN_VERBOSE) System.out.println("Generate resources.");
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (!getResource(cell).isEmpty() || getTerrain(cell) == CITY.getMapChar() || getTerrain(cell) == VILLAGE.getMapChar()) {
                continue;
            }
            boolean inner = isWithinCityInfluence(cell, 1);
            boolean outer = !inner && isWithinCityInfluence(cell, 2);
            Types.TRIBE owner = tileOwner[cell];
            char terrain = getTerrain(cell);
            if (terrain == PLAIN.getMapChar() && (inner || outer)) {
                double fruit = (inner ? 0.18 : 0.06) * getTribeProb("FRUIT", owner);
                double crop = (inner ? 0.18 : 0.06) * getTribeProb("CROPS", owner);
                placeWeightedResource(cell, fruit, FRUIT.getMapChar(), crop, CROPS.getMapChar());
            } else if (terrain == FOREST.getMapChar() && (inner || outer)) {
                double animal = (inner ? 0.19 : 0.06) * getTribeProb("ANIMAL", owner);
                if (rnd.nextDouble() < animal) {
                    writeTile(cell, null, "" + ANIMAL.getMapChar());
                }
            } else if (terrain == MOUNTAIN.getMapChar() && (inner || outer)) {
                double ore = (inner ? 0.11 : 0.03) * getTribeProb("ORE", owner);
                if (rnd.nextDouble() < ore) {
                    writeTile(cell, null, "" + ORE.getMapChar());
                }
            } else if (terrain == SHALLOW_WATER.getMapChar() && isWithinCityInfluence(cell, 2)) {
                double fish = 0.50 * getTribeProb("FISH", owner);
                if (rnd.nextDouble() < fish) {
                    writeTile(cell, null, "" + FISH.getMapChar());
                }
            } else if (terrain == DEEP_WATER.getMapChar() && isWithinCityInfluence(cell, 2)) {
                double fish = 0.25 * getTribeProb("FISH", owner);
                if (rnd.nextDouble() < fish) {
                    writeTile(cell, null, "" + FISH.getMapChar());
                }
            }
        }
    }

    private void placeWeightedResource(int cell, double firstWeight, char firstResource, double secondWeight, char secondResource) {
        double total = Math.max(0.0, firstWeight) + Math.max(0.0, secondWeight);
        double roll = rnd.nextDouble();
        if (roll < firstWeight) {
            writeTile(cell, null, "" + firstResource);
        } else if (roll < total) {
            writeTile(cell, null, "" + secondResource);
        }
    }

    private void placeRuins() {
        if (LEVELGEN_VERBOSE) System.out.println("Ruins generation");
        int ruinsNumber = getRuinCountForMapSize();
        int maxWaterRuins = mapType == Types.MAP_TYPE.LAKES ? (int) Math.floor(ruinsNumber / 3.0) : ruinsNumber;
        int waterRuins = 0;
        int ruins = 0;
        while (ruins < ruinsNumber) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int cell = 0; cell < mapSize * mapSize; cell++) {
                if (getResource(cell).isEmpty() && canPlaceRuin(cell)) {
                    boolean waterRuin = getTerrain(cell) == DEEP_WATER.getMapChar();
                    if (!waterRuin || waterRuins < maxWaterRuins) {
                        candidates.add(cell);
                    }
                }
            }
            if (candidates.isEmpty()) {
                throw new IllegalStateException("Unable to place the configured number of ruins on the generated map.");
            }
            int ruin = candidates.get(randomInt(0, candidates.size()));
            writeTile(ruin, null, "" + RUINS.getMapChar());
            if (getTerrain(ruin) == DEEP_WATER.getMapChar()) {
                waterRuins++;
            }
            ruins++;
        }
    }

    private void adjustStartingResources(ArrayList<Integer> capitalCells) {
        if (LEVELGEN_VERBOSE) System.out.println("Re-adjust starting tiles around capitals");
        for (int capital : capitalCells) {
            int owner = Integer.parseInt(getResource(capital));
            if(owner == IMPERIUS.getKey()) {
                postGenerate(FRUIT.getMapChar(), PLAIN.getMapChar(), 2, capital);
            } else if(owner == BARDUR.getKey()) {
                postGenerate(ANIMAL.getMapChar(), FOREST.getMapChar(), 2, capital);
            } else if (owner == XIN_XI.getKey()) {
                postGenerate(ORE.getMapChar(), MOUNTAIN.getMapChar(), 2, capital);
            } else if (owner == OUMAJI.getKey()) {
                postGenerate(FRUIT.getMapChar(), PLAIN.getMapChar(), 2, capital);
            } else if (owner == KICKOO.getKey()) {
                postGenerate(FISH.getMapChar(), SHALLOW_WATER.getMapChar(), 2, capital);
            }
        }
    }

    private ArrayList<Integer> villageCandidates(ArrayList<Integer> existingVillages, int edgeBuffer, int spacing) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (canPlaceVillage(cell, existingVillages, edgeBuffer, spacing)) {
                candidates.add(cell);
            }
        }
        Collections.shuffle(candidates, rnd);
        return candidates;
    }

    private boolean canPlaceVillage(int cell, ArrayList<Integer> existingVillages, int edgeBuffer, int spacing) {
        char terrain = getTerrain(cell);
        if ((terrain != PLAIN.getMapChar() && terrain != FOREST.getMapChar())
                || !getResource(cell).isEmpty()
                || nearMapEdge(cell, edgeBuffer)
                || nearAny(cell, existingVillages, spacing)
                || nearAny(cell, currentCapitalCells(), spacing)) {
            return false;
        }
        return true;
    }

    private ArrayList<Integer> preTerrainVillageCandidates(ArrayList<Integer> existingVillages, int edgeBuffer, int spacing) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (canPlacePreTerrainVillage(cell, existingVillages, edgeBuffer, spacing)) {
                candidates.add(cell);
            }
        }
        Collections.shuffle(candidates, rnd);
        return candidates;
    }

    private boolean canPlacePreTerrainVillage(int cell, ArrayList<Integer> existingVillages, int edgeBuffer, int spacing) {
        char terrain = getTerrain(cell);
        if ((terrain != DEEP_WATER.getMapChar() && terrain != PLAIN.getMapChar() && terrain != FOREST.getMapChar())
                || !getResource(cell).isEmpty()
                || nearMapEdge(cell, edgeBuffer)
                || nearAny(cell, existingVillages, spacing)
                || nearAny(cell, currentCapitalCells(), spacing)) {
            return false;
        }
        return true;
    }

    private void ensureLakesCapitalVillageConnections(ArrayList<Integer> capitalCells, ArrayList<Integer> villageCells) {
        ArrayList<Integer> allVillages = new ArrayList<>(currentVillageCells());
        for (int capital : capitalCells) {
            int attempts = 0;
            while (countReachableVillages(capital, allVillages) < 2 && attempts < mapSize * mapSize) {
                Integer extraVillage = findReachableVillageCandidate(capital, allVillages);
                if (extraVillage == null) {
                    if (!expandReachableLandmass(capital)) {
                        break;
                    }
                    normalizeDeepWaterCoastlines();
                    attempts++;
                    continue;
                }
                writeTile(extraVillage, "" + VILLAGE.getMapChar(), null);
                allVillages.add(extraVillage);
                villageCells.add(extraVillage);
                attempts++;
            }
        }
    }

    private Integer findReachableVillageCandidate(int capital, ArrayList<Integer> existingVillages) {
        boolean[] reachable = reachableLandCells(capital);
        int bestDistance = Integer.MAX_VALUE;
        ArrayList<Integer> best = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (!reachable[cell] || !canPlaceVillage(cell, existingVillages, 1, 2)) {
                continue;
            }
            int distance = distance(capital, cell, mapSize);
            if (distance < bestDistance) {
                bestDistance = distance;
                best.clear();
                best.add(cell);
            } else if (distance == bestDistance) {
                best.add(cell);
            }
        }
        if (best.isEmpty()) {
            return null;
        }
        return best.get(randomInt(0, best.size()));
    }

    private int countReachableVillages(int capital, ArrayList<Integer> villages) {
        boolean[] reachable = reachableLandCells(capital);
        int count = 0;
        for (int village : villages) {
            if (reachable[village]) {
                count++;
            }
        }
        return count;
    }

    private boolean[] reachableLandCells(int start) {
        boolean[] visited = new boolean[mapSize * mapSize];
        ArrayList<Integer> queue = new ArrayList<>();
        queue.add(start);
        visited[start] = true;
        for (int i = 0; i < queue.size(); i++) {
            int cell = queue.get(i);
            for (int neighbour : crossNeighbors(cell)) {
                if (!visited[neighbour] && !isWaterTerrain(getTerrain(neighbour))) {
                    visited[neighbour] = true;
                    queue.add(neighbour);
                }
            }
        }
        return visited;
    }

    private boolean expandReachableLandmass(int capital) {
        boolean[] reachable = reachableLandCells(capital);
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (!reachable[cell]) {
                continue;
            }
            for (int neighbour : crossNeighbors(cell)) {
                if (isWaterTerrain(getTerrain(neighbour))) {
                    candidates.add(neighbour);
                }
            }
        }
        if (candidates.isEmpty()) {
            return false;
        }
        int newLand = candidates.get(randomInt(0, candidates.size()));
        writeTile(newLand, "" + PLAIN.getMapChar(), null);
        return true;
    }

    private int findContinentSeed(int[] component) {
        for (int attempt = 0; attempt < 200; attempt++) {
            int cell = randomInteriorCell(2);
            if (component[cell] < 0 && !touchesAssignedContinent(cell, component, -1)) {
                return cell;
            }
        }
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (!nearMapEdge(cell, 2) && component[cell] < 0 && !touchesAssignedContinent(cell, component, -1)) {
                return cell;
            }
        }
        return -1;
    }

    private boolean hasContinentExpansionCell(ArrayList<Integer> frontier, int[] component, int continent) {
        for (int source : frontier) {
            for (int neighbour : crossNeighbors(source)) {
                if (canExpandContinentTo(neighbour, component, continent)) {
                    return true;
                }
            }
        }
        return false;
    }

    private Integer pickContinentExpansionCell(ArrayList<Integer> frontier, int[] component, int continent) {
        ArrayList<Integer> sources = new ArrayList<>(frontier);
        Collections.shuffle(sources, rnd);
        for (int source : sources) {
            ArrayList<Integer> candidates = new ArrayList<>(crossNeighbors(source));
            Collections.shuffle(candidates, rnd);
            for (int candidate : candidates) {
                if (canExpandContinentTo(candidate, component, continent)) {
                    return candidate;
                }
            }
        }
        return null;
    }

    private boolean canExpandContinentTo(int cell, int[] component, int continent) {
        return component[cell] < 0
                && !touchesAssignedContinent(cell, component, continent);
    }

    private boolean touchesAssignedContinent(int cell, int[] component, int allowedContinent) {
        for (int neighbour : circle(cell, 1)) {
            int neighbourComponent = component[neighbour];
            if (neighbourComponent >= 0 && neighbourComponent != allowedContinent) {
                return true;
            }
        }
        return false;
    }

    private int landComponentKey(int start) {
        boolean[] visited = new boolean[mapSize * mapSize];
        ArrayList<Integer> queue = new ArrayList<>();
        queue.add(start);
        visited[start] = true;
        int key = start;
        for (int i = 0; i < queue.size(); i++) {
            int cell = queue.get(i);
            key = Math.min(key, cell);
            for (int neighbour : circle(cell, 1)) {
                if (!visited[neighbour] && !isWaterTerrain(getTerrain(neighbour))) {
                    visited[neighbour] = true;
                    queue.add(neighbour);
                }
            }
        }
        return key;
    }

    private boolean isWaterTerrain(char terrain) {
        return terrain == SHALLOW_WATER.getMapChar() || terrain == DEEP_WATER.getMapChar();
    }

    private ArrayList<Integer> currentVillageCells() {
        ArrayList<Integer> villages = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (getTerrain(cell) == VILLAGE.getMapChar()) {
                villages.add(cell);
            }
        }
        return villages;
    }

    private ArrayList<Integer> currentCapitalCells() {
        ArrayList<Integer> capitals = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (getTerrain(cell) == CITY.getMapChar()) {
                capitals.add(cell);
            }
        }
        return capitals;
    }

    private boolean nearAny(int cell, ArrayList<Integer> others, int radius) {
        for (int other : others) {
            if (distance(cell, other, mapSize) <= radius) {
                return true;
            }
        }
        return false;
    }

    private boolean nearMapEdge(int cell, int buffer) {
        int row = cell / mapSize;
        int col = cell % mapSize;
        return row < buffer || col < buffer || row >= mapSize - buffer || col >= mapSize - buffer;
    }

    private boolean isWithinCityInfluence(int cell, int radius) {
        for (int nearby : disk(cell, radius)) {
            char terrain = getTerrain(nearby);
            if (terrain == CITY.getMapChar() || terrain == VILLAGE.getMapChar()) {
                return true;
            }
        }
        return false;
    }

    private boolean nearLand(int cell, int radius) {
        for (int nearby : disk(cell, radius)) {
            if (!isWaterTerrain(getTerrain(nearby))) {
                return true;
            }
        }
        return false;
    }

    private int randomInteriorCell(int buffer) {
        int row = randomInt(buffer, mapSize - buffer);
        int col = randomInt(buffer, mapSize - buffer);
        return row * mapSize + col;
    }

    private void forceEdgeLandBridge() {
        for (int i = 0; i < mapSize; i++) {
            writeTile(i, "" + PLAIN.getMapChar(), null);
            writeTile((mapSize - 1) * mapSize + i, "" + PLAIN.getMapChar(), null);
            writeTile(i * mapSize, "" + PLAIN.getMapChar(), null);
            writeTile(i * mapSize + mapSize - 1, "" + PLAIN.getMapChar(), null);
        }
    }

    private void forceSettlementAdjacentLand(ArrayList<Integer> capitalCells, ArrayList<Integer> villageCells) {
        ArrayList<Integer> settlements = new ArrayList<>(capitalCells);
        settlements.addAll(villageCells);
        for (int settlement : settlements) {
            if (mapType == Types.MAP_TYPE.WATER_WORLD) {
                ArrayList<Integer> candidates = new ArrayList<>();
                for (int neighbour : crossNeighbors(settlement)) {
                    if (!isSettlement(neighbour)) {
                        candidates.add(neighbour);
                    }
                }
                if (!candidates.isEmpty()) {
                    writeTile(candidates.get(randomInt(0, candidates.size())), "" + PLAIN.getMapChar(), null);
                }
                continue;
            }
            for (int nearby : circle(settlement, 1)) {
                if (!isSettlement(nearby) && isWaterTerrain(getTerrain(nearby))) {
                    writeTile(nearby, "" + PLAIN.getMapChar(), null);
                }
            }
        }
    }

    private void enforceQuadrantWaterBand() {
        double[] band = quadrantWaterBand();
        if (band == null) {
            return;
        }
        int tiles = mapSize * mapSize;
        int minWater = (int) Math.ceil(tiles * band[0]);
        int maxWater = (int) Math.floor(tiles * band[1]);
        int water = countWaterTiles();
        if (water < minWater) {
            ArrayList<Integer> candidates = mutableLandCells();
            Collections.shuffle(candidates, rnd);
            for (int cell : candidates) {
                if (water >= minWater) {
                    break;
                }
                writeTile(cell, "" + DEEP_WATER.getMapChar(), null);
                water++;
            }
        } else if (water > maxWater) {
            ArrayList<Integer> candidates = mutableWaterCells();
            Collections.shuffle(candidates, rnd);
            for (int cell : candidates) {
                if (water <= maxWater) {
                    break;
                }
                writeTile(cell, "" + PLAIN.getMapChar(), null);
                water--;
            }
        }
    }

    private double[] quadrantWaterBand() {
        switch (mapType) {
            case DRYLANDS:
                return new double[]{0.0, 0.0};
            case LAKES:
                return new double[]{0.25, 0.30};
            case ARCHIPELAGO:
                return new double[]{0.60, 0.80};
            case WATER_WORLD:
                return new double[]{0.90, 1.00};
            default:
                return null;
        }
    }

    private int countWaterTiles() {
        int water = 0;
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (isWaterTerrain(getTerrain(cell))) {
                water++;
            }
        }
        return water;
    }

    private ArrayList<Integer> mutableLandCells() {
        ArrayList<Integer> cells = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            int row = cell / mapSize;
            int col = cell % mapSize;
            if (!isSettlement(cell)
                    && !isWaterTerrain(getTerrain(cell))
                    && (!profile.edgeLandBridge || !isVillageEdgeRestrictedTile(row, col))
                    && !isCriticalSettlementAdjacentLand(cell)) {
                cells.add(cell);
            }
        }
        return cells;
    }

    private ArrayList<Integer> mutableWaterCells() {
        ArrayList<Integer> cells = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (!isSettlement(cell) && getResource(cell).isEmpty() && isWaterTerrain(getTerrain(cell))) {
                cells.add(cell);
            }
        }
        return cells;
    }

    private boolean isSettlement(int cell) {
        char terrain = getTerrain(cell);
        return terrain == CITY.getMapChar() || terrain == VILLAGE.getMapChar();
    }

    private boolean isCriticalSettlementAdjacentLand(int cell) {
        for (int settlement : crossNeighbors(cell)) {
            if (!isSettlement(settlement)) {
                continue;
            }
            int adjacentLand = 0;
            for (int neighbour : crossNeighbors(settlement)) {
                if (!isWaterTerrain(getTerrain(neighbour))) {
                    adjacentLand++;
                }
            }
            if (adjacentLand <= 1) {
                return true;
            }
        }
        return false;
    }

    private int quadrantDomainRows() {
        return tribes.length <= 4 ? 2 : tribes.length <= 9 ? 3 : 4;
    }

    private int quadrantDomainCols() {
        return quadrantDomainRows();
    }

    private ArrayList<Integer> quadrantDomainCells(int domain, ArrayList<Integer> existingCapitals) {
        int rows = quadrantDomainRows();
        int cols = quadrantDomainCols();
        int regionRow = domain / cols;
        int regionCol = domain % cols;
        int usableRows = Math.max(1, mapSize - 4);
        int usableCols = Math.max(1, mapSize - 4);
        int minRow = 2 + regionRow * usableRows / rows;
        int maxRow = regionRow == rows - 1 ? mapSize - 2 : 2 + (regionRow + 1) * usableRows / rows;
        int minCol = 2 + regionCol * usableCols / cols;
        int maxCol = regionCol == cols - 1 ? mapSize - 2 : 2 + (regionCol + 1) * usableCols / cols;
        ArrayList<Integer> cells = new ArrayList<>();
        for (int row = minRow; row < maxRow; row++) {
            for (int col = minCol; col < maxCol; col++) {
                int cell = row * mapSize + col;
                if (!existingCapitals.contains(cell)) {
                    cells.add(cell);
                }
            }
        }
        return cells;
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private void configureMapType(Types.MAP_TYPE mapType) {
        switch (mapType) {
            case DRYLANDS:
                this.smoothing = 2;
                this.relief = 4;
                this.initialLand = 0.88;
                this.BORDER_EXPANSION = 0.20;
                break;
            case LAKES:
                this.smoothing = 4;
                this.relief = 4;
                this.initialLand = 0.72;
                this.BORDER_EXPANSION = 0.33;
                break;
            case PANGEA:
                this.smoothing = 4;
                this.relief = 4;
                this.initialLand = 0.78;
                this.BORDER_EXPANSION = 0.33;
                break;
            case ARCHIPELAGO:
                this.smoothing = 2;
                this.relief = 3;
                this.initialLand = 0.45;
                this.BORDER_EXPANSION = 0.40;
                break;
            case WATER_WORLD:
                this.smoothing = 1;
                this.relief = 3;
                this.initialLand = 0.28;
                this.BORDER_EXPANSION = 0.50;
                break;
            case CONTINENTS:
            default:
                this.smoothing = 3;
                this.relief = 4;
                this.initialLand = 0.58;
                this.BORDER_EXPANSION = 1 / 3.0;
                break;
        }
        this.landCoefficient = (0.5 + relief) / 9;
    }

    private ArrayList<Integer> selectQuadrantCapitalsFromDomains() {
        int domainCount = quadrantDomainRows() * quadrantDomainCols();
        int trials = Math.max(250, domainCount * tribes.length * 20);
        CapitalLayout best = null;
        for (int trial = 0; trial < trials; trial++) {
            CapitalLayout layout = randomCapitalLayout(domainCount);
            if (layout != null && (best == null || layout.score > best.score)) {
                best = layout;
            }
        }
        if (best != null) {
            return best.capitals;
        }
        return fallbackQuadrantCapitals(domainCount);
    }

    private CapitalLayout randomCapitalLayout(int domainCount) {
        ArrayList<Integer> domains = new ArrayList<>();
        for (int domain = 0; domain < domainCount; domain++) {
            domains.add(domain);
        }
        Collections.shuffle(domains, rnd);
        ArrayList<Integer> capitals = new ArrayList<>();
        ArrayList<Integer> chosenDomains = new ArrayList<>();
        for (int seat = 0; seat < tribes.length; seat++) {
            if (domains.isEmpty()) {
                return null;
            }
            int domain = domains.remove(0);
            ArrayList<Integer> candidates = quadrantDomainCells(domain, capitals);
            if (candidates.isEmpty()) {
                return null;
            }
            capitals.add(weightedRandomCapital(candidates, capitals));
            chosenDomains.add(domain);
        }
        return new CapitalLayout(capitals, scoreCapitalLayout(capitals, chosenDomains));
    }

    private ArrayList<Integer> fallbackQuadrantCapitals(int domainCount) {
        ArrayList<Integer> capitals = new ArrayList<>();
        ArrayList<Integer> domains = new ArrayList<>();
        for (int domain = 0; domain < domainCount; domain++) {
            domains.add(domain);
        }
        Collections.shuffle(domains, rnd);
        for (int i = 0; i < tribes.length; i++) {
            int domain = domains.isEmpty() ? randomInt(0, domainCount) : domains.remove(0);
            ArrayList<Integer> cells = quadrantDomainCells(domain, capitals);
            capitals.add(cells.isEmpty() ? randomInteriorCell(2) : pickBestCapital(cells, capitals, false));
        }
        return capitals;
    }

    private int weightedRandomCapital(ArrayList<Integer> candidates, ArrayList<Integer> existingCapitals) {
        double total = 0.0;
        double[] weights = new double[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            weights[i] = capitalSoftWeight(candidates.get(i), existingCapitals);
            total += weights[i];
        }
        double roll = rnd.nextDouble() * total;
        for (int i = 0; i < candidates.size(); i++) {
            roll -= weights[i];
            if (roll <= 0.0) {
                return candidates.get(i);
            }
        }
        return candidates.get(candidates.size() - 1);
    }

    private double scoreCapitalLayout(ArrayList<Integer> capitals, ArrayList<Integer> domains) {
        int minDistance = mapSize;
        int distanceTotal = 0;
        int pairs = 0;
        for (int i = 0; i < capitals.size(); i++) {
            for (int j = i + 1; j < capitals.size(); j++) {
                int distance = distance(capitals.get(i), capitals.get(j), mapSize);
                minDistance = Math.min(minDistance, distance);
                distanceTotal += distance;
                pairs++;
            }
        }
        if (pairs == 0) {
            minDistance = mapSize;
        }

        int edgeTotal = 0;
        int edgeMin = mapSize;
        int localSpaceTotal = 0;
        for (int capital : capitals) {
            int row = capital / mapSize;
            int col = capital % mapSize;
            int edgeDistance = Math.min(Math.min(row, mapSize - 1 - row), Math.min(col, mapSize - 1 - col));
            edgeTotal += edgeDistance;
            edgeMin = Math.min(edgeMin, edgeDistance);
            localSpaceTotal += countDomainCellsNear(capital, 2);
        }

        int domainSpread = domainSpreadScore(domains);
        int spacingPenalty = Math.max(0, minimumCapitalDistance() - minDistance) * 10000;
        return minDistance * 1000
                + distanceTotal * 10
                + edgeMin * 80
                + edgeTotal * 8
                + localSpaceTotal * 6
                + domainSpread * 25
                - spacingPenalty;
    }

    private int countDomainCellsNear(int center, int radius) {
        int count = 0;
        for (int cell : disk(center, radius)) {
            if (!nearMapEdge(cell, 1)) {
                count++;
            }
        }
        return count;
    }

    private int domainSpreadScore(ArrayList<Integer> domains) {
        int score = 0;
        int cols = quadrantDomainCols();
        for (int i = 0; i < domains.size(); i++) {
            int rowA = domains.get(i) / cols;
            int colA = domains.get(i) % cols;
            for (int j = i + 1; j < domains.size(); j++) {
                int rowB = domains.get(j) / cols;
                int colB = domains.get(j) % cols;
                score += Math.max(Math.abs(rowA - rowB), Math.abs(colA - colB));
            }
        }
        return score;
    }

    private int minimumCapitalDistance() {
        return Math.max(4, mapSize / 4);
    }

    private ArrayList<Integer> filterByMinimumCapitalDistance(ArrayList<Integer> candidates,
                                                               ArrayList<Integer> existingCapitals) {
        int minimumCapitalDistance = minimumCapitalDistance();
        ArrayList<Integer> filtered = new ArrayList<>();
        for (Integer candidate : candidates) {
            if (!nearAny(candidate, existingCapitals, minimumCapitalDistance - 1)) {
                filtered.add(candidate);
            }
        }
        return filtered;
    }

    private double capitalSoftWeight(int cell, ArrayList<Integer> existingCapitals) {
        int row = cell / mapSize;
        int col = cell % mapSize;
        int edgeDistance = Math.min(Math.min(row, mapSize - 1 - row), Math.min(col, mapSize - 1 - col));
        int capitalDistance = mapSize;
        for (int capital : existingCapitals) {
            capitalDistance = Math.min(capitalDistance, distance(cell, capital, mapSize));
        }
        double edgeWeight = 1.0 + Math.max(0, edgeDistance - 1) * 0.35;
        double distanceWeight = existingCapitals.isEmpty() ? 1.0 : Math.max(1.0, capitalDistance);
        return edgeWeight * distanceWeight;
    }

    private int pickBestCapital(ArrayList<Integer> candidates, ArrayList<Integer> existingCapitals, boolean preferCoastal) {
        if (preferCoastal) {
            ArrayList<Integer> coastalCandidates = new ArrayList<>();
            for (Integer candidate : candidates) {
                if (isCoastalCell(candidate)) {
                    coastalCandidates.add(candidate);
                }
            }
            if (!coastalCandidates.isEmpty()) {
                candidates = coastalCandidates;
            }
        }

        int bestScore = Integer.MIN_VALUE;
        ArrayList<Integer> best = new ArrayList<>();
        for (Integer candidate : candidates) {
            int score = scoreCapitalCandidate(candidate, existingCapitals, preferCoastal);
            if (score > bestScore) {
                bestScore = score;
                best.clear();
                best.add(candidate);
            } else if (score == bestScore) {
                best.add(candidate);
            }
        }

        if (best.isEmpty()) {
            return candidates.get(randomInt(0, candidates.size()));
        }
        return best.get(randomInt(0, best.size()));
    }

    private int pickBestVillageCapital(ArrayList<Integer> candidates, ArrayList<Integer> existingCapitals, boolean preferCoastal) {
        ArrayList<Integer> filtered = filterByMinimumCapitalDistance(candidates, existingCapitals);
        if (!filtered.isEmpty()) {
            candidates = filtered;
        }
        return pickBestCapital(candidates, existingCapitals, preferCoastal);
    }

    private int scoreCapitalCandidate(int candidate, ArrayList<Integer> existingCapitals, boolean preferCoastal) {
        int minDistance = mapSize;
        for (Integer capital : existingCapitals) {
            minDistance = Math.min(minDistance, distance(candidate, capital, mapSize));
        }
        if (existingCapitals.isEmpty()) {
            minDistance = mapSize;
        }

        int coastBonus = preferCoastal && isCoastalCell(candidate) ? mapSize / 2 : 0;
        int centerPenalty = mapType == Types.MAP_TYPE.WATER_WORLD ? distance(candidate, ((mapSize / 2) * mapSize) + (mapSize / 2), mapSize) : 0;
        return minDistance * 10 + coastBonus - centerPenalty;
    }

    private boolean isCoastalCell(int cell) {
        for (int neighbour : crossNeighbors(cell)) {
            char terrain = getTerrain(neighbour);
            if (terrain == SHALLOW_WATER.getMapChar() || terrain == DEEP_WATER.getMapChar()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Counts the instances of a resource that exists on the starting tiles that surround a capital.
     * @param resource the resource to be counted.
     * @param capital the index of the capital.
     * @return the resource counter.
     */
    public int checkResources(char resource, int capital) {
        int resources = 0;
        for (int neighbour : circle(capital, 1)) {
            String resourceStr = getResource(neighbour);
            if(resourceStr.length() > 0 && resourceStr.charAt(0) == resource){
                resources++;
            }
        }
        return resources;
    }

    /**
     * Adds the required amount of a specific resource on specific type of terrain
     * in the starting tiles that surround a capital.
     * @param resource the resource to be added.
     * @param terrain the terrain on top of which the resource will be added.
     * @param quantity the amount to be tiles that must have this terrain + resource combination.
     */
    public void postGenerate(char resource, char terrain, int quantity, int capital) {
        int resources = checkResources(resource, capital);
        int attempts = 0;
        while (resources < quantity && attempts < mapSize * mapSize) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int tile : circle(capital, 1)) {
                char currentTerrain = getTerrain(tile);
                String currentResource = getResource(tile);
                if (currentTerrain != CITY.getMapChar()
                        && currentTerrain != VILLAGE.getMapChar()
                        && (!isWaterTerrain(currentTerrain) || terrain == SHALLOW_WATER.getMapChar())
                        && (currentResource.isEmpty() || currentResource.charAt(0) != resource)) {
                    candidates.add(tile);
                }
            }
            if (candidates.isEmpty()) {
                return;
            }
            int target = candidates.get(randomInt(0, candidates.size()));
            writeTile(target, ""+terrain, ""+resource);
            for (int neighbour : crossNeighbors(target)) {
                if (getTerrain(neighbour) == DEEP_WATER.getMapChar() && getResource(neighbour).isEmpty()) {
                    writeTile(neighbour, ""+SHALLOW_WATER.getMapChar(), null);
                }
            }
            resources = checkResources(resource, capital);
            attempts++;
        }
    }
    /**
     * Utility function used in the generator.
     */
    public boolean proc(ArrayList<Integer> villageMap, int cell, double probability) {
        return (villageMap.get(cell) == 2 && rnd.nextDouble() < probability) || (villageMap.get(cell) == 1 && rnd.nextDouble() < probability * BORDER_EXPANSION);
    }

    private void placeCornerLighthouses() {
        int[] corners = new int[]{
                0,
                mapSize - 1,
                (mapSize - 1) * mapSize,
                mapSize * mapSize - 1
        };

        for (int corner : corners) {
            char terrain = getTerrain(corner);
            if (terrain == MOUNTAIN.getMapChar() || terrain == FOREST.getMapChar()) {
                terrain = PLAIN.getMapChar();
            }
            writeTile(corner, "" + terrain, "" + Types.RESOURCE.LIGHTHOUSE.getMapChar());
        }
    }

    private void normalizeDeepWaterCoastlines() {
        ArrayList<Integer> coastalDeepWater = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            if (getTerrain(cell) != DEEP_WATER.getMapChar()) {
                continue;
            }
            for (int neighbour : crossNeighbors(cell)) {
                char terrain = getTerrain(neighbour);
                if (terrain != DEEP_WATER.getMapChar() && terrain != SHALLOW_WATER.getMapChar()) {
                    coastalDeepWater.add(cell);
                    break;
                }
            }
        }

        for (int cell : coastalDeepWater) {
            writeTile(cell, "" + SHALLOW_WATER.getMapChar(), null);
        }
    }

    private int getRuinCountForMapSize() {
        switch (mapSize) {
            case 11:
                return 4;
            case 14:
                return 5;
            case 16:
                return 7;
            case 18:
                return 9;
            case 20:
                return 11;
            case 30:
                return 23;
            default:
                throw new IllegalStateException("Unsupported map size side length: " + mapSize);
        }
    }

    private boolean isVillageEdgeRestrictedTile(int row, int column) {
        return row == 0 || row == mapSize - 1 || column == 0 || column == mapSize - 1;
    }

    private boolean canPlaceRuin(int cell) {
        char terrain = getTerrain(cell);
        if (terrain != PLAIN.getMapChar()
                && terrain != FOREST.getMapChar()
                && terrain != MOUNTAIN.getMapChar()
                && terrain != DEEP_WATER.getMapChar()) {
            return false;
        }
        for (int nearby : disk(cell, 1)) {
            if (getTerrain(nearby) == VILLAGE.getMapChar()
                    || getTerrain(nearby) == CITY.getMapChar()
                    || getResource(nearby).equals("" + RUINS.getMapChar())) {
                return false;
            }
        }
        return true;
    }

    private void placeStarfish() {
        ArrayList<Integer> waterCandidates = new ArrayList<>();
        for (int cell = 0; cell < mapSize * mapSize; cell++) {
            char terrain = getTerrain(cell);
            if ((terrain == DEEP_WATER.getMapChar() || terrain == SHALLOW_WATER.getMapChar())
                    && getResource(cell).isEmpty()) {
                waterCandidates.add(cell);
            }
        }

        Collections.shuffle(waterCandidates, rnd);
        int targetStarfish = countWaterTiles() / 25;
        int placedStarfish = 0;

        for (int candidate : waterCandidates) {
            if (placedStarfish >= targetStarfish) {
                break;
            }
            if (canPlaceStarfish(candidate)) {
                writeTile(candidate, null, "" + STARFISH.getMapChar());
                placedStarfish++;
            }
        }
    }

    private boolean canPlaceStarfish(int cell) {
        char terrain = getTerrain(cell);
        if (!getResource(cell).isEmpty()
                || (terrain != DEEP_WATER.getMapChar() && terrain != SHALLOW_WATER.getMapChar())) {
            return false;
        }

        for (int neighbour : circle(cell, 1)) {
            String resource = getResource(neighbour);
            if (!resource.isEmpty()) {
                Types.RESOURCE neighbourResource = Types.RESOURCE.getType(resource.charAt(0));
                if (neighbourResource == STARFISH || neighbourResource == Types.RESOURCE.LIGHTHOUSE) {
                    return false;
                }
            }

            if (getTerrain(neighbour) == CITY.getMapChar()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Reads the JSON configuration file and returns the probability of a terrain or a resource for a specific tribe.
     * @param name the name of the terrain or resource.
     * @param tribe the name of the tribe.
     * @return the probability.
     */
    public double getTribeProb(String name, Types.TRIBE tribe) {
        if(tribe == null) {
            return 1.0;
        } else {
            return data.getJSONObject(name).getDouble(tribe.toString());
        }
    }

    /**
     * Reads the JSON configuration file and returns the base probability of a specific terrain or resource.
     * @param name the name of the terrain or resource.
     * @return the base probability.
     */
    public double getBaseProb(String name) {
        return data.getJSONObject(name).getDouble("BASE");
    }

    /**
     * Writes a level tile at a specified position (consult the TERRAIN and RESOURCE enums).
     * @param index the index of the tile that needs to be written.
     * @param terrain the desired type of terrain.
     * @param resource the desired type of resource.
     */
    public void writeTile(int index, String terrain, String resource) {
        if(terrain == null) {
            level[index] = "" + getTerrain(index) + ':' + resource;
        }else if(resource == null) {
            level[index] = "" + terrain + ':' + getResource(index);
        }else {
            level[index] = "" + terrain + ':' + resource;
        }
    }

    /**
     * Returns a tile's terrain at a specified position.
     * @param index the desired position.
     * @return the character that represents the specific terrain (consult TERRAIN enum).
     */
    public char getTerrain(int index) {
        return level[index].split(":")[0].charAt(0);
    }


//    public char getResource(int index) {
//        return level[index].split(":")[1].charAt(0);
////        try {
////            return level[index].split(":")[1].charAt(0);
////        } catch(Exception e) {
////            return '';
////        }
//    }
    /**
     * Returns a tile's resource at a specified position.
     * @param index the desired position.
     * @return the character that represents the specific resource (consult RESOURCE enum).
     */
    public String getResource(int index)
    {
        String[] pieces = level[index].split(":", -1);
        if(pieces.length > 1)
            if(pieces[1].isEmpty() || pieces[1].charAt(0) == ' ')
                return "";
            else return pieces[1];
        else return "";
    }

    /**
     * Returns a random int in the range [min, max).
     * @param min lower bound (inclusive).
     * @param max upper bound (exclusive).
     * @return a random int.
     */
    public int randomInt(int min, int max) {
        return (int) Math.floor(min + rnd.nextDouble() * (max - min));
    }

    /**
     * Returns the indices of the map that lie on a circle.
     * @param center center of the circle.
     * @param radius radius of the circle.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> circle(int center, int radius) {
        ArrayList<Integer> circle = new ArrayList<>();
        int row = center / mapSize;
        int column = center % mapSize;
        int i = row - radius;
        if (i >= 0 && i < mapSize) {
            for (int j = column - radius; j < column + radius; j++) {
                if (j >= 0 && j < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        i = row + radius;
        if (i >= 0 && i < mapSize) {
            for (int j = column + radius; j > column - radius; j--) {
                if (j >= 0 && j < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        int j = column - radius;
        if (j >= 0 && j < mapSize) {
            for (i = row + radius; i > row - radius; i--) {
                if (i >= 0 && i < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        j = column + radius;
        if (j >= 0 && j < mapSize) {
            for (i = row - radius; i < row + radius; i++) {
                if (i >= 0 && i < mapSize) {
                    circle.add(i * mapSize + j);
                }
            }
        }
        return circle;
    }

    /**
     * Returns the indices of the map that lie on and inside a circle including the center.
     * @param center center of the circle.
     * @param radius radius of the circle.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> disk(int center, int radius) {
        ArrayList<Integer> round = new ArrayList<>();
        for (int r = 1; r <= radius; r++) {
            round.addAll(circle(center, r));
        }
        round.add(center);
        return round;
    }

    /**
     * Returns the indices of the map that lie on the cross pattern.
     * @param center center of the cross.
     * @return an ArrayList of indices.
     */
    public ArrayList<Integer> crossNeighbors(int center) {
        ArrayList<Integer> plus_sign = new ArrayList<>();
        int row = center / mapSize;
        int column = center % mapSize;
        if (column > 0) {
            plus_sign.add(center - 1);
        }
        if (column < mapSize - 1) {
            plus_sign.add(center + 1);
        }
        if (row > 0) {
            plus_sign.add(center - mapSize);
        }
        if (row < mapSize - 1) {
            plus_sign.add(center + mapSize);
        }
        return plus_sign;
    }

    // we use pythagorean distances
    public int distance(int a, int b, int size) {
        int ax = a % size;
        int ay = a / size;
        int bx = b % size;
        int by = b / size;
        return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
    }

    private static final class GenerationProfile {
        private final double landRatio;
        private final int smoothing;
        private final double landCoefficient;
        private final boolean edgeLandBridge;
        private final boolean suburbs;
        private final double preTerrainVillageCoefficient;
        private final int postTerrainVillageEdgeBuffer;
        private final boolean capitalsFromVillages;
        private final boolean coastalCapitals;
        private final int tinyIslandVillages;

        private GenerationProfile(double landRatio, int smoothing, double landCoefficient, boolean edgeLandBridge,
                                  boolean suburbs, double preTerrainVillageCoefficient, int postTerrainVillageEdgeBuffer,
                                  boolean capitalsFromVillages, boolean coastalCapitals, int tinyIslandVillages) {
            this.landRatio = landRatio;
            this.smoothing = smoothing;
            this.landCoefficient = landCoefficient;
            this.edgeLandBridge = edgeLandBridge;
            this.suburbs = suburbs;
            this.preTerrainVillageCoefficient = preTerrainVillageCoefficient;
            this.postTerrainVillageEdgeBuffer = postTerrainVillageEdgeBuffer;
            this.capitalsFromVillages = capitalsFromVillages;
            this.coastalCapitals = coastalCapitals;
            this.tinyIslandVillages = tinyIslandVillages;
        }

        private static GenerationProfile forMapType(Types.MAP_TYPE mapType, int mapSize) {
            int islandVillages = tinyIslandVillagesForSize(mapSize);
            switch (mapType) {
                case DRYLANDS:
                    return new GenerationProfile(0.96, 1, 0.70, false, false, 0.0, 3, false, false, 0);
                case LAKES:
                    return new GenerationProfile(0.50, 3, 0.48, true, true, 0.3, 3, false, false, 0);
                case PANGEA:
                    return new GenerationProfile(0.52, 0, 0.0, false, false, 0.0, 3, true, true, islandVillages);
                case ARCHIPELAGO:
                    return new GenerationProfile(0.40, 1, 0.45, false, true, 0.3, 3, false, true, 0);
                case WATER_WORLD:
                    return new GenerationProfile(0.10, 1, 0.28, false, false, 0.1, 3, false, true, 0);
                case CONTINENTS:
                default:
                    return new GenerationProfile(0.50, 3, 0.52, false, false, 0.0, 3, true, false, islandVillages);
            }
        }

        private static int tinyIslandVillagesForSize(int mapSize) {
            switch (mapSize) {
                case 11:
                    return 0;
                case 14:
                    return 1;
                case 16:
                    return 2;
                case 18:
                    return 3;
                case 20:
                    return 4;
                case 30:
                    return 9;
                default:
                    return 0;
            }
        }
    }

    private static final class TerrainRates {
        private final double mountain;
        private final double forest;

        private TerrainRates(double mountain, double forest) {
            this.mountain = mountain;
            this.forest = forest;
        }
    }

    private static final class CapitalLayout {
        private final ArrayList<Integer> capitals;
        private final double score;

        private CapitalLayout(ArrayList<Integer> capitals, double score) {
            this.capitals = capitals;
            this.score = score;
        }
    }

    /**
     * Saves the generated level into a .csv format readable by the Tribes framework.
     * @param filename path to save the level.
     */
    public void toCSV(String filename) {
        try {
            FileWriter writer = new FileWriter(filename);
            writer.append(tileForOutput(0));
            writer.append(',');
            for(int i = 1; i < mapSize*mapSize; i++) {
                if(i % mapSize == 0) {
                    writer.append('\n');
                    writer.append(tileForOutput(i));
                    writer.append(',');
                } else if(i % mapSize == mapSize - 1) {
                    writer.append(tileForOutput(i));
                }else {
                    writer.append(tileForOutput(i));
                    writer.append(',');
                }
            }
            writer.flush();
            writer.close();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to write generated level to '" + filename + "'.", e);
        }
    }

    /**
     * Prints the generated level in console.
     */
    public void print() {
        StringBuffer writer = new StringBuffer();
        writer.append(tileForOutput(0));
        writer.append(',');
        for (int i = 1; i < mapSize * mapSize; i++) {
            if (i % mapSize == 0) {
                writer.append('\n');
                writer.append(tileForOutput(i));
                writer.append(',');
            } else if (i % mapSize == mapSize - 1) {
                writer.append(tileForOutput(i));
            } else {
                writer.append(tileForOutput(i));
                writer.append(',');
            }
        }
        System.out.println(writer.toString());
    }

    /**
     * Returns the generated level into a format readable by the Tribes framework.
     */
    public String[] gelLevelLines()
    {
        String[] allLines = new String[mapSize];
        int lineCounter = 0;

        StringBuffer line = new StringBuffer();
        line.append(tileForOutput(0));
        line.append(',');
        for (int i = 1; i < mapSize * mapSize; i++) {
            if (i % mapSize == mapSize - 1) {
                line.append(tileForOutput(i));
                allLines[lineCounter] = line.toString();
                lineCounter++;
                line = new StringBuffer();
            } else {
                line.append(tileForOutput(i));
                line.append(',');
            }
        }
        return allLines;
    }

    public static void main(String[] args) {

        long genSeed = System.currentTimeMillis();
        LevelGenerator gen = new LevelGenerator(genSeed);
        gen.init(Types.MAP_SIZE.TINY.getSideLength(), Types.MAP_TYPE.CONTINENTS, new Types.TRIBE[]{XIN_XI, OUMAJI});
        gen.generate();
        gen.toCSV("levels/levelgen_test.csv");
        gen.print();
        
    }
}
