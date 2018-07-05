import logging

from jmetal.algorithm import SMPSO
from jmetal.component.observer import ProgressBarObserver, VisualizerObserver
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.problem import ZDT1
from jmetal.operator import Polynomial
from jmetal.util.graphic import ScatterMatplotlib
from jmetal.util.solution_list_output import SolutionList


def main() -> None:
    problem = ZDT1()

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100)
    )

    observer = VisualizerObserver(problem)
    progress_bar = ProgressBarObserver(step=100, max=25000)
    algorithm.observable.register(observer=observer)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    result = algorithm.get_result()

    # Plot frontier to file
    pareto_front = ScatterMatplotlib(plot_title='SMPSO for ZDT1', number_of_objectives=problem.number_of_objectives)
    pareto_front.plot(result, reference=problem.get_reference_front(), output='SMPSO-ZDT1', show=False)

    # Save variables to file
    SolutionList.print_function_values_to_file(result, 'SMPSO.ZDT1')

    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Computing time: " + str(algorithm.total_computing_time))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler('jmetalpy.log'),
            logging.StreamHandler()
        ]
    )

    main()