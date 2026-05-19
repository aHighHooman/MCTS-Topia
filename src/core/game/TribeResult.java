package core.game;

import core.Types;

public class TribeResult implements Comparable<TribeResult>
{
    private Types.RESULT result;
    private int id;
    private int score;
    private int numTechsResearched;
    private int numCities;
    private int production;

    public TribeResult(int id, Types.RESULT res, int score, int numTechsResearched, int numCities, int production)
    {
        this.id = id;
        this.result = res;
        this.score = score;
        this.numTechsResearched = numTechsResearched;
        this.numCities = numCities;
        this.production = production;
    }

    @Override
    public int compareTo(TribeResult other)
    {
        //Winning status determines
        if(this.result == Types.RESULT.WIN && other.result != Types.RESULT.WIN)
            return -1;
        else if (this.result != Types.RESULT.WIN && other.result == Types.RESULT.WIN)
            return 1;

        //Tie breaker 0: score
        if(this.score > other.score)
            return -1;
        else if (this.score < other.score)
            return 1;


        //Tie breaker 1: num tech researched
        if(this.numTechsResearched > other.numTechsResearched)
            return -1;
        else if (this.numTechsResearched < other.numTechsResearched)
            return 1;

        //Tie breaker 2: num cities owned
        if(this.numCities > other.numCities)
            return -1;
        else if(this.numCities < other.numCities)
            return 1;

        //Tie breaker 3: production
        if(this.production > other.production)
            return -1;
        else if(this.production < other.production)
            return 1;

        //Keep fully tied results stable so ranking order is reproducible.
        return Integer.compare(this.id, other.id);
    }

    public int getId() {
        return id;
    }

    public int getNumTechsResearched() {
        return numTechsResearched;
    }

    public int getNumCities() {
        return numCities;
    }

    public int getProduction() {
        return production;
    }

    public double getScore() {
        return score;
    }

    public Types.RESULT getResult() {
        return result;
    }

    public TribeResult copy() {
        return new TribeResult(id, result, score, numTechsResearched, numCities, production);
    }

    public void setResult(Types.RESULT result) {
        this.result = result;
    }
}
