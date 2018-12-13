import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
from scipy import stats

from jmetal.component.quality_indicator import QualityIndicator
from jmetal.core.algorithm import Algorithm
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_solutions

LOGGER = logging.getLogger('jmetal')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Job:

    def __init__(self, algorithm: Algorithm, algorithm_tag: str, run: int):
        self.algorithm = algorithm
        self.algorithm_tag = algorithm_tag
        self.run_tag = run

    def execute(self, path: str):
        self.algorithm.run()

        file_name = os.path.join(path, 'FUN.{}.ps'.format(self.run_tag))
        print_function_values_to_file(self.algorithm.get_result(), file_name=file_name)

        file_name = os.path.join(path, 'VAR.{}.ps'.format(self.run_tag))
        print_variables_to_file(self.algorithm.get_result(), file_name=file_name)


class Experiment:

    def __init__(self, base_directory: str, jobs: List[Job], m_workers: int = 3):
        """ Run an experiment to evaluate algorithms and/or problems.

        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.jobs = jobs
        self.m_workers = m_workers
        self.base_directory = base_directory

    def run(self) -> None:
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                executor.submit(job.execute(self.base_directory))


def compute_quality_indicator(input_data: str, quality_indicators: List[QualityIndicator],
                              reference_fronts: str = '') -> None:
    """ Compute a list of quality indicators. The input data directory *must* met the following structure (this is generated
    automatically by the Experiment class):

    * <base_dir>

      * algorithm_a

        * problem_a

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

        * problem_b

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

      * algorithm_b

        * ...

    For each indicator a new file `QI.<name_of_the_indicator>` is created inside each problem folder, containing the values computed for each front.

    :param input_data: Directory where all the input data is found (function values and variables).
    :param reference_fronts: Directory where reference fronts are found.
    :param quality_indicators: List of quality indicators to compute.
    :return: None.
    """
    for dirname, _, filenames in os.walk(input_data):
        for filename in filenames:
            algorithm, problem = dirname.split('/')[-2:]

            if 'FUN' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))

                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + '.pf')

                    # Add reference front if found
                    if hasattr(indicator, 'reference_front'):
                        if Path(reference_front_file).is_file():
                            indicator.reference_front = read_solutions(reference_front_file)
                        else:
                            LOGGER.warning('Reference front not found at', reference_front_file)

                    # Save quality indicator value to file
                    # Note: We need to ensure that the result is inserted at the correct row inside the file
                    with open('{}/QI.{}'.format(dirname, indicator.get_name()), 'a+') as of:
                        index = [int(s) for s in filename.split('.') if s.isdigit()].pop()

                        contents = of.readlines()
                        contents.insert(index, str(indicator.compute(solutions)) + '\n')

                        of.seek(0)  # readlines consumes the iterator, so we need to start over
                        of.writelines(contents)


def create_tables_from_experiment(input_data: str):
    pd.set_option('display.float_format', '{:.2e}'.format)
    df = pd.DataFrame()

    for dirname, _, filenames in os.walk(input_data):
        for filename in filenames:
            algorithm, problem = dirname.split('/')[-2:]

            if 'QI' in filename:
                with open(os.path.join(dirname, filename), 'r+') as of:
                    contents = of.readlines()

                    for index, value in enumerate(contents):
                        new_data = pd.DataFrame({
                            'problem': problem,
                            'run': index,
                            filename: [float(value)]
                        })
                        df = df.append(new_data)

    # Get rid of NaN values by grouping rows by columns
    df = df.groupby(['problem', 'run']).mean()

    return df


def compute_statistical_analysis(df: pd.DataFrame):
    """ The application scheme listed here is as described in

    * G. Luque, E. Alba, Parallel Genetic Algorithms, Springer-Verlag, ISBN 978-3-642-22084-5, 2011

    :param df: Experiment data frame.
    """
    if len(df.columns) < 2:
        raise Exception('Data sets number must be equal or greater than two')

    result = pd.DataFrame()

    # we assume non-normal variables (median comparison, non-parametric tests)
    if len(df.columns) == 2:
        LOGGER.info('Running non-parametric test: Wilcoxon signed-rank test')
        for _, subset in df.groupby(level=0):
            statistic, pvalue = stats.wilcoxon(subset[subset.columns[0]], subset[subset.columns[1]])

            test = pd.DataFrame({
                'Wilcoxon': '*' if pvalue < 0.05 else '-'
            }, index=[subset.index.values[0][0]], columns=['Wilcoxon'])
            test.index.name = 'problem'

            result = result.append(test)
    else:
        LOGGER.info('Running non-parametric test: Kruskal-Wallis test')
        for _, subset in df.groupby(level=0):
            statistic, pvalue = stats.kruskal(*subset.values.tolist())

            test = pd.DataFrame({
                'Kruskal-Wallis': '*' if pvalue < 0.05 else '-'
            }, index=[subset.index.values[0][0]], columns=['Kruskal-Wallis'])
            test.index.name = 'problem'

            result = result.append(test)

    return result


def convert_to_latex(df: pd.DataFrame, caption: str, label: str = 'tab:exp', alignment: str = 'c'):
    """ Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold, does not use math mode.
    """
    num_columns, num_rows = df.shape[1], df.shape[0]
    output = io.StringIO()

    col_format = '{}|{}'.format(alignment, alignment * num_columns)
    column_labels = ['\\textbf{{{0}}}'.format(label.replace('_', '\\_')) for label in df.columns]

    # Write header
    output.write('\\begin{table}\n')
    output.write('\\caption{{{}}}\n'.format(caption))
    output.write('\\label{{{}}}\n'.format(label))
    output.write('\\centering\n')
    output.write('\\begin{scriptsize}\n')
    output.write('\\begin{tabular}{%s}\n' % col_format)
    output.write('\\hline\n')
    output.write('& {} \\\\\\hline\n'.format(' & '.join(column_labels)))

    # Write data lines
    for i in range(num_rows):
        output.write('\\textbf{{{0}}} & ${1}$ \\\\\n'.format(
            df.index[i], '$ & $'.join([str(val) for val in df.ix[i]]))
        )

    # Write footer
    output.write('\\hline\n')
    output.write('\\end{tabular}\n')
    output.write('\\end{scriptsize}\n')
    output.write('\\end{table}')

    return output.getvalue()
