#!/usr/bin/env python3
"""
PS12 - Assignment 2: Bayesian Network

Implements exact inference for two binary Bayesian Networks using complete
joint enumeration.

Usage:
    python PS12.py
    python PS12.py [input_file]
    python PS12.py [input_file] [output_file]

Default files:
    inputPS12.txt
    outputPS12.txt
"""

import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

# ==============================================================================
# Global Constants & Configuration
# ==============================================================================
SCENARIO_1 = "SCENARIO_1_ROAD_ACCIDENT"
SCENARIO_2 = "SCENARIO_2_TRAFFIC_SIGNAL_FAILURE"

# Required keys for Scenario 1 (Road Accident: A -> D, A -> E)
REQUIRED_1 = (
    "P_A",
    "P_D_given_A",
    "P_D_given_notA",
    "P_E_given_A",
    "P_E_given_notA",
)

# Required keys for Scenario 2 (Signal Failure: S -> C, S -> R)
REQUIRED_2 = (
    "P_S",
    "P_C_given_S",
    "P_C_given_notS",
    "P_R_given_S",
    "P_R_given_notS",
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================
class InputError(Exception):
    """Raised when the assignment input file is invalid or malformed."""

    pass


# ==============================================================================
# Core Bayesian Network Exact Inference Engine
# ==============================================================================
class BinaryBayesianNetwork:
    """
    Reusable exact-inference engine for binary Bayesian Networks using complete
    joint probability table enumeration.
    """

    def __init__(self, nodes, parents, true_prob_fn):
        """
        Initializes the binary Bayesian Network engine.

        Args:
            nodes (iterable): Variable names in topological order.
            parents (dict): Map of variable name to tuple of parent variable names.
            true_prob_fn (callable): Callable taking (node, assignment) returning
                P(node=True | parents).
        """
        self.nodes = tuple(nodes)
        self.parents = dict(parents)
        self.true_prob_fn = true_prob_fn

    def joint_probability(self, assignment):
        """
        Calculates P(X1, X2, ..., Xn) using Bayesian Network factorization rule:
        P(X1, X2, ..., Xn) = Product of P(Xi | Parents(Xi)).

        Args:
            assignment (dict): Map of all variable names to Boolean values.

        Returns:
            float: Calculated joint probability for the specified assignment.
        """
        p = 1.0
        for node in self.nodes:
            p_true = self.true_prob_fn(node, assignment)
            p *= p_true if assignment[node] else (1.0 - p_true)
        return p

    def enumerate_joint(self):
        """
        Generates the complete Joint Probability Table (2^N rows).

        Returns:
            list of tuple: List containing (assignment_dict, probability_float) entries.
        """
        rows = []
        for values in product((True, False), repeat=len(self.nodes)):
            assignment = dict(zip(self.nodes, values))
            rows.append((assignment, self.joint_probability(assignment)))
        return rows

    def probability_of_evidence(self, evidence):
        """
        Marginalizes the joint table by summing probabilities of all rows matching evidence.

        Args:
            evidence (dict): Map of observed variable names to Boolean values.

        Returns:
            float: Total marginal probability P(evidence).
        """
        total = 0.0
        for assignment, p in self.enumerate_joint():
            if all(assignment[k] == v for k, v in evidence.items()):
                total += p
        return total

    def query(self, query_var, evidence):
        """
        Computes exact posterior conditional probability P(query_var=True | evidence).

        Args:
            query_var (str): Query variable name.
            evidence (dict): Observed evidence variable assignments.

        Returns:
            float: Conditional probability P(query_var=True | evidence).

        Raises:
            ZeroDivisionError: If P(evidence) is zero.
        """
        if query_var in evidence:
            return 1.0 if evidence[query_var] else 0.0

        denominator = self.probability_of_evidence(evidence)
        if denominator == 0.0:
            raise ZeroDivisionError(
                f"Evidence {evidence} has probability 0; posterior is undefined."
            )

        numerator_evidence = dict(evidence)
        numerator_evidence[query_var] = True
        numerator = self.probability_of_evidence(numerator_evidence)

        return numerator / denominator

    def are_independent(self, x, y, tolerance=1e-12):
        """
        Verifies marginal independence: P(X=True, Y=True) == P(X=True) * P(Y=True).

        Args:
            x (str): Name of first variable.
            y (str): Name of second variable.
            tolerance (float, optional): Absolute tolerance threshold. Defaults to 1e-12.

        Returns:
            bool: True if variables are marginally independent within tolerance.
        """
        px = self.probability_of_evidence({x: True})
        py = self.probability_of_evidence({y: True})
        pxy = self.probability_of_evidence({x: True, y: True})
        return abs(pxy - px * py) <= tolerance

    def are_conditionally_independent(self, x, y, given, tolerance=1e-12):
        """
        Numerically verifies conditional independence X ⟂ Y | Z across all possible
        state combinations of the conditioning variables Z.

        Args:
            x (str): Name of first variable.
            y (str): Name of second variable.
            given (iterable): Sequence of conditioning variable names.
            tolerance (float, optional): Absolute tolerance threshold. Defaults to 1e-12.

        Returns:
            bool: True if conditionally independent for all Z state combinations.
        """
        given = tuple(given)

        for values in product((True, False), repeat=len(given)):
            evidence = dict(zip(given, values))
            pz = self.probability_of_evidence(evidence)
            if pz == 0.0:
                continue

            pxy_given_z = (
                self.probability_of_evidence({**evidence, x: True, y: True}) / pz
            )
            px_given_z = self.probability_of_evidence({**evidence, x: True}) / pz
            py_given_z = self.probability_of_evidence({**evidence, y: True}) / pz

            if abs(pxy_given_z - px_given_z * py_given_z) > tolerance:
                return False

        return True


# ==============================================================================
# Bayesian Network Scenario Builders
# ==============================================================================
def build_scenario_1(values):
    """
    Constructs Scenario 1 Network (Road Accident Prediction):
    A (Road Accident) -> D (Traffic Delay)
    A (Road Accident) -> E (Emergency Call)

    Args:
        values (dict): Parameter values parsed from input file for Scenario 1.

    Returns:
        BinaryBayesianNetwork: Instantiated network object for Scenario 1.
    """
    def p_true(node, assignment):
        """Calculates P(node=True | parents) for Scenario 1 variables."""
        if node == "A":
            return values["P_A"]
        if node == "D":
            return values["P_D_given_A"] if assignment["A"] else values["P_D_given_notA"]
        if node == "E":
            return values["P_E_given_A"] if assignment["A"] else values["P_E_given_notA"]
        raise KeyError(f"Unknown node: {node}")

    return BinaryBayesianNetwork(
        nodes=("A", "D", "E"),
        parents={"A": (), "D": ("A",), "E": ("A",)},
        true_prob_fn=p_true,
    )


def build_scenario_2(values):
    """
    Constructs Scenario 2 Network (Traffic Signal Failure Detection):
    S (Signal Failure) -> C (Camera Alert)
    S (Signal Failure) -> R (Sensor Alert)

    Args:
        values (dict): Parameter values parsed from input file for Scenario 2.

    Returns:
        BinaryBayesianNetwork: Instantiated network object for Scenario 2.
    """
    def p_true(node, assignment):
        """Calculates P(node=True | parents) for Scenario 2 variables."""
        if node == "S":
            return values["P_S"]
        if node == "C":
            return values["P_C_given_S"] if assignment["S"] else values["P_C_given_notS"]
        if node == "R":
            return values["P_R_given_S"] if assignment["S"] else values["P_R_given_notS"]
        raise KeyError(f"Unknown node: {node}")

    return BinaryBayesianNetwork(
        nodes=("S", "C", "R"),
        parents={"S": (), "C": ("S",), "R": ("S",)},
        true_prob_fn=p_true,
    )


# ==============================================================================
# Input File Parser
# ==============================================================================
def parse_input(path):
    """
    Parses and strictly validates the assignment input file format.

    Args:
        path (str or Path): Input file path to parse.

    Returns:
        dict: Parsed nested dictionary containing scenario key-value probabilities.

    Raises:
        InputError: If file is missing, empty, malformed, or contains invalid probabilities.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"Input file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [(i + 1, line.strip()) for i, line in enumerate(raw_lines)]
    lines = [(n, s) for n, s in lines if s]

    if not lines:
        raise InputError("Input file is empty.")

    first_line_no, first = lines[0]
    if first != "PS12":
        raise InputError(f"Line {first_line_no}: expected 'PS12', found {first!r}.")

    data = {SCENARIO_1: {}, SCENARIO_2: {}}
    current = None
    seen_scenarios = set()

    for line_no, line in lines[1:]:
        if line in (SCENARIO_1, SCENARIO_2):
            if line in seen_scenarios:
                raise InputError(f"Line {line_no}: duplicate scenario identifier {line!r}.")
            current = line
            seen_scenarios.add(line)
            continue

        if line.startswith("SCENARIO_"):
            raise InputError(f"Line {line_no}: invalid scenario identifier {line!r}.")

        if current is None:
            raise InputError(f"Line {line_no}: probability appears before a scenario header.")

        if "=" not in line or line.count("=") != 1:
            raise InputError(f"Line {line_no}: malformed input line {line!r}; expected key=value.")

        key, value_text = [part.strip() for part in line.split("=", 1)]
        if not key or not value_text:
            raise InputError(f"Line {line_no}: malformed key/value in {line!r}.")

        if key in data[current]:
            raise InputError(f"Line {line_no}: duplicate key {key!r} in {current}.")

        try:
            value = float(value_text)
        except ValueError as exc:
            raise InputError(f"Line {line_no}: {value_text!r} is not a valid probability.") from exc

        if not (0.0 <= value <= 1.0):
            raise InputError(f"Line {line_no}: probability {key}={value} must be between 0 and 1.")

        data[current][key] = value

    missing_scenarios = {SCENARIO_1, SCENARIO_2} - seen_scenarios
    if missing_scenarios:
        raise InputError("Missing scenario identifier(s): " + ", ".join(sorted(missing_scenarios)))

    for scenario, required in ((SCENARIO_1, REQUIRED_1), (SCENARIO_2, REQUIRED_2)):
        missing = [k for k in required if k not in data[scenario]]
        extra = [k for k in data[scenario] if k not in required]

        if missing:
            raise InputError(f"{scenario}: missing required key(s): {', '.join(missing)}")
        if extra:
            raise InputError(f"{scenario}: unexpected key(s): {', '.join(extra)}")

    return data


def tf(value):
    """
    Converts a Boolean value to single-character 'T' or 'F' string.

    Args:
        value (bool): Input Boolean value.

    Returns:
        str: 'T' if True, 'F' if False.
    """
    return "T" if value else "F"


def fmt4(value):
    """
    Formats a floating point number or Decimal to 4 decimal places with standard ROUND_HALF_UP.

    Args:
        value (float or Decimal or str): Number to format.

    Returns:
        str: Formatted numeric string rounded to 4 decimal places.
    """
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def make_output(data):
    """
    Executes all Bayesian Network inference queries and constructs formatted output text.

    Args:
        data (dict): Parsed input dictionary containing probabilities for both scenarios.

    Returns:
        str: Formatted output string ready to write to output file.
    """
    bn1 = build_scenario_1(data[SCENARIO_1])
    bn2 = build_scenario_2(data[SCENARIO_2])

    lines = []

    # --------------------------------------------------------------------------
    # Scenario 1 Output
    # --------------------------------------------------------------------------
    lines.append("Scenario 1 - Joint Probability Table")
    lines.append("")
    lines.append("A   D   E   Probability")
    for assignment, p in bn1.enumerate_joint():
        lines.append(
            f"{tf(assignment['A'])}   "
            f"{tf(assignment['D'])}   "
            f"{tf(assignment['E'])}   "
            + fmt4(p)
        )

    p_a_given_d = bn1.query("A", {"D": True})
    p_a_given_e = bn1.query("A", {"E": True})
    p_a_given_de = bn1.query("A", {"D": True, "E": True})

    de_independent = bn1.are_independent("D", "E")
    de_cond_independent = bn1.are_conditionally_independent("D", "E", given=("A",))

    lines.append("")
    lines.append("Scenario 1: Road Accident Prediction")
    lines.append(f"P(A | D) = {fmt4(p_a_given_d)}")
    lines.append(f"P(A | E) = {fmt4(p_a_given_e)}")
    lines.append(f"P(A | D and E) = {fmt4(p_a_given_de)}")
    lines.append(
        "Traffic Delay and Emergency Call are "
        + ("" if de_independent else "not ")
        + "independent without evidence."
    )
    lines.append(
        "Traffic Delay and Emergency Call are "
        + ("" if de_cond_independent else "not ")
        + "conditionally independent given Road Accident."
    )

    # --------------------------------------------------------------------------
    # Scenario 2 Output
    # --------------------------------------------------------------------------
    p_s_given_c = bn2.query("S", {"C": True})
    p_s_given_r = bn2.query("S", {"R": True})
    p_s_given_cr = bn2.query("S", {"C": True, "R": True})

    cr_independent = bn2.are_independent("C", "R")
    cr_cond_independent = bn2.are_conditionally_independent("C", "R", given=("S",))

    lines.append("")
    lines.append("Scenario 2: Traffic Signal Failure Detection")
    lines.append(f"P(S | C) = {fmt4(p_s_given_c)}")
    lines.append(f"P(S | R) = {fmt4(p_s_given_r)}")
    lines.append(f"P(S | C and R) = {fmt4(p_s_given_cr)}")
    lines.append(
        "Camera Alert and Sensor Alert are "
        + ("" if cr_independent else "not ")
        + "independent without evidence."
    )
    lines.append(
        "Camera Alert and Sensor Alert are "
        + ("" if cr_cond_independent else "not ")
        + "conditionally independent given Traffic Signal Failure."
    )

    return "\n".join(lines) + "\n"


# ==============================================================================
# Development / Testing Helper Functions (Commented Out for Submission)
# ==============================================================================
#
# def test_print_cpt(bn, network_name="Network"):
#     """
#     Utility function used during development to verify CPT values
#     and joint probability enumeration table output.
#     """
#     print(f"=== {network_name} Joint Probability Table ===")
#     for assignment, prob in bn.enumerate_joint():
#         states = ", ".join(f"{k}={v}" for k, v in assignment.items())
#         print(f"P({states}) = {prob:.6f}")
#     print()


# ==============================================================================
# CLI Entry Point & Main Execution Block
# ==============================================================================
def main():
    """
    Main CLI entry point managing input/output paths, execution, and exception handling.

    Returns:
        int: Status code (0 for success, 1 for execution error, 2 for CLI usage error).
    """
    input_path = sys.argv[1] if len(sys.argv) >= 2 else "inputPS12.txt"
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "outputPS12.txt"

    if len(sys.argv) > 3:
        print(
            "Usage: python PS12.py [input_file] [output_file]",
            file=sys.stderr,
        )
        return 2

    try:
        data = parse_input(input_path)
        output = make_output(data)
        Path(output_path).write_text(output, encoding="utf-8")
        print(f"Output written to {output_path}")
        return 0
    except (InputError, ZeroDivisionError, OSError) as exc:
        error_message = f"ERROR: {exc}"
        print(error_message, file=sys.stderr)
        try:
            Path(output_path).write_text(error_message + "\n", encoding="utf-8")
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
