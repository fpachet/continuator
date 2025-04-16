import numpy as np
from collections import namedtuple


class NoSolutionError(Exception):
    def __init__(self, message):
        self.message = message


# code by Jessica Stringham:
# https://jessicastringham.net/2019/01/09/sum-product-message-passing/

LabeledArray = namedtuple(
    "LabeledArray",
    [
        "array",
        "axes_labels",
    ],
)


def name_to_axis_mapping(labeled_array):
    return {name: axis for axis, name in enumerate(labeled_array.axes_labels)}


def other_axes_from_labeled_axes(labeled_array, axis_label):
    # returns the indexes of the axes that are not axis label
    return tuple(
        axis
        for axis, name in enumerate(labeled_array.axes_labels)
        if name != axis_label
    )


def is_conditional_prob(labeled_array, var_name):
    """
    labeled_array (LabeledArray)
    variable (str): name of variable, i.e. 'a' in p(a|b)
    """
    return np.all(
        np.isclose(
            np.sum(
                labeled_array.array, axis=name_to_axis_mapping(labeled_array)[var_name]
            ),
            1.0,
        )
    )


def is_joint_prob(labeled_array):
    return np.all(np.isclose(np.sum(labeled_array.array), 1.0))


def tile_to_shape_along_axis(arr, target_shape, target_axis):
    # get a list of all axes
    raw_axes = list(range(len(target_shape)))
    tile_dimensions = [target_shape[a] for a in raw_axes if a != target_axis]
    if len(arr.shape) == 0:
        # If given a scalar, also tile it in the target dimension (so it's a bunch of 1s)
        tile_dimensions += [target_shape[target_axis]]
    elif len(arr.shape) == 1:
        # If given an array, it should be the same shape as the target axis
        assert arr.shape[0] == target_shape[target_axis]
        tile_dimensions += [1]
    else:
        raise NotImplementedError()
    tiled = np.tile(arr, tile_dimensions)

    # Tiling only adds prefix axes, so rotate this one back into place
    shifted_axes = raw_axes[:target_axis] + [raw_axes[-1]] + raw_axes[target_axis:-1]
    transposed = np.transpose(tiled, shifted_axes)

    # Double-check this code tiled it to the correct shape
    assert transposed.shape == target_shape
    return transposed


def tile_to_other_dist_along_axis_name(tiling_labeled_array, target_array):
    assert len(tiling_labeled_array.axes_labels) == 1
    target_axis_label = tiling_labeled_array.axes_labels[0]

    return LabeledArray(
        tile_to_shape_along_axis(
            tiling_labeled_array.array,
            target_array.array.shape,
            name_to_axis_mapping(target_array)[target_axis_label],
        ),
        axes_labels=target_array.axes_labels,
    )


class Node(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def __repr__(self):
        return "{classname}({name}, [{neighbors}])".format(
            classname=type(self).__name__,
            name=self.name,
            neighbors=", ".join([n.name for n in self.neighbors]),
        )

    def is_valid_neighbor(self, neighbor):
        raise NotImplemented()

    def add_neighbor(self, neighbor):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)


class Variable(Node):
    def is_valid_neighbor(self, factor):
        return isinstance(factor, Factor)  # Variables can only neighbor Factors


class Factor(Node):
    def is_valid_neighbor(self, variable):
        return isinstance(variable, Variable)  # Factors can only neighbor Variables

    def __init__(self, name):
        super(Factor, self).__init__(name)
        self.data = None


ParsedTerm = namedtuple(
    "ParsedTerm",
    [
        "term",
        "var_name",
        "given",
    ],
)


def _parse_term(term):
    # Given a term like (a|b,c), returns a list of variables
    # and conditioned-on variables
    assert term[0] == "(" and term[-1] == ")"
    term_variables = term[1:-1]

    # Handle conditionals
    if "|" in term_variables:
        var, given = term_variables.split("|")
        given = given.split(",")
    else:
        var = term_variables
        given = []

    return var, given


def _parse_model_string_into_terms(model_string):
    return [
        ParsedTerm("p" + term, *_parse_term(term))
        for term in model_string.split("p")
        if term
    ]


def parse_model_into_variables_and_factors(model_string):
    # Takes in a model_string such as p(h1)p(h2∣h1)p(v1∣h1)p(v2∣h2) and returns a
    # dictionary of variable names to variables and a list of factors.

    # Split model_string into ParsedTerms
    parsed_terms = _parse_model_string_into_terms(model_string)

    # First, extract all of the variables from the model_string (h1, h2, v1, v2).
    # These each will be a new Variable that are referenced from Factors below.
    variables = {}
    for parsed_term in parsed_terms:
        # if the variable name wasn't seen yet, add it to the variables dict
        if parsed_term.var_name not in variables:
            variables[parsed_term.var_name] = Variable(parsed_term.var_name)

    # Now extract factors from the model. Each term (e.g. "p(v1|h1)") corresponds to
    # a factor.
    # Then find all variables in this term ("v1", "h1") and add the corresponding Variables
    # as neighbors to the new Factor, and this Factor to the Variables' neighbors.
    factors = []
    for parsed_term in parsed_terms:
        # This factor will be neighbors with all "variables" (left-hand side variables) and given variables
        new_factor = Factor(parsed_term.term)
        all_var_names = [parsed_term.var_name] + parsed_term.given
        for var_name in all_var_names:
            new_factor.add_neighbor(variables[var_name])
            variables[var_name].add_neighbor(new_factor)
        factors.append(new_factor)

    return factors, variables


class PGM(object):
    def __init__(self, factors, variables):
        self._factors = factors
        self._variables = variables

    @classmethod
    def from_string(cls, model_string):
        factors, variables = parse_model_into_variables_and_factors(model_string)
        return PGM(factors, variables)

    def set_data(self, data):
        # Keep track of variable dimensions to check for shape mistakes
        var_dims = {}
        for factor in self._factors:
            factor_data = data[factor.name]

            if set(factor_data.axes_labels) != set(v.name for v in factor.neighbors):
                missing_axes = set(v.name for v in factor.neighbors) - set(
                    data[factor.name].axes_labels
                )
                raise ValueError(
                    "data[{}] is missing axes: {}".format(factor.name, missing_axes)
                )

            for var_name, dim in zip(factor_data.axes_labels, factor_data.array.shape):
                if var_name not in var_dims:
                    var_dims[var_name] = dim

                if var_dims[var_name] != dim:
                    raise ValueError(
                        "data[{}] axes is wrong size, {}. Expected {}".format(
                            factor.name, dim, var_dims[var_name]
                        )
                    )

            factor.data = data[factor.name]

    def variable_from_name(self, var_name):
        return self._variables[var_name]

    def factor_from_name(self, fac_name):
        for f in self._factors:
            if f.name == fac_name:
                return f
        print(f"factor not found: {fac_name}")
        return None

    def print_marginals(self):
        for var in self._variables.values():
            print(f"marginal: {var.name}: {Messages().marginal(var)}")

    def set_value(self, var_name, value_idx):
        factor = self.factor_from_name('p(' + var_name + ')')
        data = np.zeros(len(factor.data.array))
        data[value_idx] = 1
        factor.data = LabeledArray(data, [var_name])


class Messages(object):
    def __init__(self):
        self.messages = {}

    def _variable_to_factor_messages(self, variable, factor):
        # print (f"_variable_to_factor_messages: {variable} to {factor}")        # Take the product over all incoming factors into this variable except the variable
        incoming_messages = [
            self.factor_to_variable_message(neighbor_factor, variable)
            for neighbor_factor in variable.neighbors
            if neighbor_factor.name != factor.name
        ]

        # If there are no incoming messages, this is 1
        return np.prod(incoming_messages, axis=0)

    def _factor_to_variable_messages(self, factor, variable):
        # print (f"_factor_to_variable_message: {factor} to {variable}")
        # Compute the product
        factor_dist = np.copy(factor.data.array)
        for neighbor_variable in factor.neighbors:
            if neighbor_variable.name == variable.name:
                continue
            incoming_message = self.variable_to_factor_messages(
                neighbor_variable, factor
            )
            factor_dist *= tile_to_other_dist_along_axis_name(
                LabeledArray(incoming_message, [neighbor_variable.name]), factor.data
            ).array
        # Sum over the axes that aren't `variable`
        other_axes = other_axes_from_labeled_axes(factor.data, variable.name)
        return np.squeeze(np.sum(factor_dist, axis=other_axes))

    def marginal(self, variable):
        # p(variable) is proportional to the product of incoming messages to variable.
        unnorm_p = np.prod(
            [
                self.factor_to_variable_message(neighbor_factor, variable)
                for neighbor_factor in variable.neighbors
            ],
            axis=0,
        )
        # At this point, we can normalize this distribution
        somme = np.sum(unnorm_p)
        if somme == 0:
            raise NoSolutionError("marginals are nan")
        return unnorm_p / somme

    def variable_to_factor_messages(self, variable, factor):
        # print (f"variable_to_factor_messages: {variable} to {factor}")
        message_name = (variable.name, factor.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._variable_to_factor_messages(
                variable, factor
            )
        return self.messages[message_name]

    def factor_to_variable_message(self, factor, variable):
        # print (f"factor_to_variable_message: {factor} to {variable}")
        message_name = (factor.name, variable.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._factor_to_variable_messages(
                factor, variable
            )
        return self.messages[message_name]


def one_hot(size, index):
    data = np.zeros(size, dtype=float)
    data[index] = 1
    return data
