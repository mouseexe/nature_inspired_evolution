import math
from collections import namedtuple
import random as r
import numpy

Parameters = namedtuple('Parameters', 'popSize, scaleFactor, crossoverRate terminationCondition')
p = Parameters(500, 0.4, 0.5, 35)

Plant = namedtuple('Plant', 'kwhPerPlant costPerPlant maxPlants')
plant1 = Plant(50000, 10000, 100)
plant2 = Plant(600000, 80000, 50)
plant3 = Plant(4000000, 400000, 3)

Market = namedtuple('Market', 'maxPrice maxDemand')
market1 = Market(0.45, 2000000)
market2 = Market(0.25, 30000000)
market3 = Market(0.2, 20000000)

def cost(x, kwhPerPlant, costPerPlant, maxPlants):
  
  if x <= 0:
    return 0

  if x > kwhPerPlant * maxPlants:
    return 999999999999999

  plantsNeeded = math.ceil(x / kwhPerPlant)

  return plantsNeeded * costPerPlant


def demand(price, maxPrice, maxDemand):

  if price > maxPrice:
    return 0

  if price <= 0:
    return maxDemand

  demand = maxDemand - price**2 * maxDemand / maxPrice**2

  return demand


def profit(solution):
  (e1, e2, e3, s1, s2, s3, p1, p2, p3) = solution

  revenue = 0
  revenue += min(demand(p1, market1.maxPrice, market1.maxDemand), s1) * p1
  revenue += min(demand(p2, market2.maxPrice, market2.maxDemand), s2) * p2
  revenue += min(demand(p3, market3.maxPrice, market3.maxDemand), s3) * p3

  productionCost = 0
  productionCost += cost(e1, plant1.kwhPerPlant, plant1.costPerPlant, plant1.maxPlants)
  productionCost += cost(e2, plant2.kwhPerPlant, plant2.costPerPlant, plant2.maxPlants)
  productionCost += cost(e3, plant3.kwhPerPlant, plant3.costPerPlant, plant3.maxPlants)

  purchasingCost = max(((s1 + s2 + s3) - (e1 + e2 + e3)), 0) * 0.6

  totalCost = productionCost + purchasingCost

  totalProfit = revenue - totalCost

  return int(totalProfit)


def cleanSolution(solution):
  cleaned = tuple(int(solution[i]*100)/100 if i > 5 else int(solution[i]) for i in range(len(solution)))
  return cleaned


def initialization(p):
  maxEnergy1 = plant1.kwhPerPlant * plant1.maxPlants
  maxEnergy2 = plant2.kwhPerPlant * plant2.maxPlants
  maxEnergy3 = plant3.kwhPerPlant * plant3.maxPlants

  solutions = []
  for i in range(p.popSize):
    e1 = r.randint(0, maxEnergy1)
    e2 = r.randint(0, maxEnergy2)
    e3 = r.randint(0, maxEnergy3)

    p1 = r.randint(0, market1.maxPrice * 100) / 100
    p2 = r.randint(0, market2.maxPrice * 100) / 100
    p3 = r.randint(0, market3.maxPrice * 100) / 100

    s1 = int(market1.maxDemand - p1**2 * market1.maxDemand / market1.maxPrice**2)
    s2 = int(market2.maxDemand - p2**2 * market2.maxDemand / market2.maxPrice**2)
    s3 = int(market3.maxDemand - p3**2 * market3.maxDemand / market3.maxPrice**2)

    solutions.append((e1, e2, e3, s1, s2, s3, p1, p2, p3))
  return solutions


def generateDonor(solutions, best, p):
  vectors = r.sample(range(0, len(solutions)), 2)
  base = best
  shift = tuple(x*p.scaleFactor for x in numpy.subtract(solutions[vectors[0]], solutions[vectors[1]]))
  donor = tuple(numpy.add(base, shift))
  return cleanSolution(donor)


def generateTrial(current, donor, p):
  donorIdx = r.randint(0, len(current))
  return tuple(donor[i] if r.random() < p.crossoverRate or i == donorIdx else current[i] for i in range(len(current)))


def selection(current, trial):
  if profit(current) > profit(trial):
    return current
  return trial


def main():
  solutions = initialization(p)
  noImprovement = 0
  globalBest = profit(solutions[0])
  while noImprovement < p.terminationCondition:
    
    best = profit(solutions[0])
    bestSolution = solutions[0]
    for solution in solutions:
      if profit(solution) > best:
        best = profit(solution)
        bestSolution = solution

    if best == globalBest:
      noImprovement += 1
    else:
      noImprovement = 0
      globalBest = best

    newSolutions = []
    for solution in solutions:
      donor = generateDonor(solutions, bestSolution, p)
      trial = generateTrial(solution, donor, p)
      newSolutions.append(selection(solution, trial))
    solutions = newSolutions
  print(best)
  print(bestSolution)


main()
