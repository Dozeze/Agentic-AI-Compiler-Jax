import timeit
import statistics
import jax

def timeFunction(function, args, numRuns = 30, numTests = 5):
    """ Compiles and runs python script, and then times it """
    """ Returns mean and standard deviation """

    times = []
    for _ in range(numTests):
        _ = function(*args).block_until_ready() #Compile first

        time = timeit.timeit(
            lambda : function(*args).block_until_ready(),
            number = numRuns
        )

        avg_time = time / numRuns

        times.append(avg_time)

    
    mean = sum(times) / numTests
    std = statistics.stdev(times)

    return mean, std
