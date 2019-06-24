import math
from collections import namedtuple
import random as r
import numpy
import csv

# Define a parameters tuple
Parameters = namedtuple('Parameters', 'popSize, scaleFactor, crossoverRate terminationCondition')

# Change parameters here
########################################
globalCost = 0.6
outputFile = 'results.csv'

populations = [50, 100, 500, 1000, 2500]
scaleFactors = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
crossoverRates = [.25, .5, .75]
repeatParams = 3
terminationCondition = 35

# Define a plant tuple and array of plants
Plant = namedtuple('Plant', 'kwhPerPlant costPerPlant maxPlants')
plant1 = Plant(50000, 10000, 100)
plant2 = Plant(600000, 80000, 50)
plant3 = Plant(4000000, 400000, 3)
plants = [plant1, plant2, plant3]

# Define a market tuple and array of markets
Market = namedtuple('Market', 'maxPrice maxDemand')
market1 = Market(0.45, 1000000)
market2 = Market(0.25, 20000000)
market3 = Market(0.2, 30000000)
markets = [market1, market2, market3]
########################################


# cost method from slides
def cost(x, kwhPerPlant, costPerPlant, maxPlants):
  
  if x <= 0:
    return 0

  if x > kwhPerPlant * maxPlants:
    return 999999999999999

  plantsNeeded = math.ceil(x / kwhPerPlant)

  return plantsNeeded * costPerPlant


# demand method from slides
def demand(price, maxPrice, maxDemand):

  if price > maxPrice:
    return 0

  if price <= 0:
    return maxDemand

  demand = maxDemand - price**2 * maxDemand / maxPrice**2

  return demand


# Calculate profit of an individual solution
def profit(s):

  # Sum the revenue from each market
  revenue = 0
  for i in range(len(markets)):
    revenue += min(demand(s[i+6], markets[i].maxPrice, markets[i].maxDemand), s[i+3]) * s[i+6]

  # Sum the production cost from each plant
  productionCost = 0
  for i in range(len(plants)):
    productionCost += cost(s[i], plants[i].kwhPerPlant, plants[i].costPerPlant, plants[i].maxPlants)

  # Calculate how much energy must be purchased from other companies
  purchasingCost = max(((s[3] + s[4] + s[5]) - (s[0] + s[1] + s[2])), 0) * globalCost

  # Calculate and return profit
  totalCost = productionCost + purchasingCost
  totalProfit = revenue - totalCost
  return int(totalProfit)


# Remove extra decimals from an individual solution
def cleanSolution(solution):

  # Convert energy values to ints to remove decimals
  # Multiply prices by 100, then convert to int before dividing by 100 to round to two decimal places
  cleaned = tuple(int(solution[i]*100)/100 if i > 5 else int(solution[i]) for i in range(len(solution)))

  # Negative values should be converted to 0 to avoid overflows
  cleaned = tuple(0 if x < 0 else x for x in cleaned)

  # Values above max should be reduced to max values
  cleaned = tuple(
      plants[i].kwhPerPlant * plants[i].maxPlants if i < 3 and plants[i].kwhPerPlant * plants[i].maxPlants < cleaned[i] 
      else markets[i%3].maxDemand if i >= 3 and i < 6 and markets[i%3].maxDemand < cleaned[i] 
      else markets[i%3].maxPrice if i >= 6 and markets[i%3].maxPrice < cleaned[i] 
      else cleaned[i] for i in range(len(cleaned)))

  return cleaned


# random initialization, with a few restrictions to ensure solutions are feasible
def initialization(p):

  # Calculate max amount of energy from each plant type
  maxEnergies = [plant.kwhPerPlant * plant.maxPlants for plant in plants]

  # Generate a full population of solutions
  solutions = []
  for i in range(p.popSize):
    # Generate energy production between 0 and max energy production
    produced = [r.randint(0, maxEnergy) for maxEnergy in maxEnergies]

    # Generate sale prices between 0 and max market price
    price = [r.randint(0, int(market.maxPrice * 100)) / 100 for market in markets]

    # Generate sale amount between 0 and max market demand
    sold = [r.randint(0, int(markets[i].maxDemand - price[i]**2 * markets[i].maxDemand / markets[i].maxPrice**2)) for i in range(len(markets))]

    # Add solution to solutions array
    solutions.append((produced[0], produced[1], produced[2], sold[0], sold[1], sold[2], price[0], price[1], price[2]))
  return solutions


# Basic donor generation, k = 1 and using best solution as a base
def generateDonor(solutions, best, p):

  # Retrieve two distinct random solutions
  vectors = r.sample(range(0, len(solutions)), 2)

  # Use best for base vector
  base = best

  # Subtract the two random vectors, then multiply by the scale factor
  shift = tuple(x*p.scaleFactor for x in numpy.subtract(solutions[vectors[0]], solutions[vectors[1]]))

  # Add shift vector to donor vector
  donor = tuple(numpy.add(base, shift))
  return cleanSolution(donor)


# Basic trial generation, using a binomial crossover
def generateTrial(current, donor, p):

  # Define a value to be guaranteed to be taken from donor
  donorIdx = r.randint(0, len(current))

  # Take from donor if r[0,1] is below crossover rate, otherwise take from current vector
  return tuple(donor[i] if r.random() < p.crossoverRate or i == donorIdx else current[i] for i in range(len(current)))


# Basic selection, add new solution if it is better than old
def selection(current, trial):

  # If current vector has a better profit than trial, return current
  if profit(current) > profit(trial):
    return current
  return trial


def main(p):

  # Initialize solutions with parameters
  solutions = initialization(p)

  # Count iterations without improved best solution
  noImprovement = 0

  # Count total runs
  runCount = 0

  # Set a baseline best solution
  globalBest = profit(solutions[0])

  # Loop until we reach enough iterations without any improvement
  while noImprovement < p.terminationCondition:
    
    # Increment total runs for statistics
    runCount += 1

    # Output status every 50 iterations
    #if runCount % 50 == 0:
    #  print('Iteration:', runCount)
    #  print('Current best profit:', globalBest)

    # Set local best solution in current pool
    best = profit(solutions[0])
    bestSolution = solutions[0]

    # Check for better best solution in current pool
    for solution in solutions:
      if profit(solution) > best:
        best = profit(solution)
        bestSolution = solution

    # Compare local best to global best
    if best == globalBest:
      noImprovement += 1
    else:
      noImprovement = 0
      globalBest = best

    # Create new pool
    newSolutions = []

    # Operate on each solution in pool
    for solution in solutions:
      # Generate donor vector with current best solution
      donor = generateDonor(solutions, bestSolution, p)

      # Generate trial with donor vector
      trial = generateTrial(solution, donor, p)

      # Add better solution to new pool
      newSolutions.append(selection(solution, trial))

    # Replace old pool with new pool
    solutions = newSolutions

  # Add end of iteration, return evalutation
  return [p.popSize, p.scaleFactor, p.crossoverRate, runCount, best, bestSolution]
  #print('Algorithm complete!')
  #print('Total iterations:', runCount)
  #print('Total profit:', best)
  #print('Best solution:', bestSolution)
  #print('Population size:', p.popSize)
  #print('Scale factor:', p.scaleFactor)
  #print('Crossover rate:', p.crossoverRate)


with open(outputFile, 'a') as csvFile:
  writer = csv.writer(csvFile)
  writer.writerow(['Population Size', 'Scale Factor', 'Crossover Rate', 'Total Iterations', 'Total Profit', 'Solution'])

  for population in populations:
    for scaleFactor in scaleFactors:
      for crossoverRate in crossoverRates:
        p = Parameters(population, scaleFactor, crossoverRate, terminationCondition)
        for i in range(repeatParams):
          writer.writerow(main(p))
csvFile.close()
