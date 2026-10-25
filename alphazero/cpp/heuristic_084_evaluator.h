#ifndef JUNQI_AZ_HEURISTIC_084_EVALUATOR_H
#define JUNQI_AZ_HEURISTIC_084_EVALUATOR_H

namespace junqi::az {

// Static evaluator extracted from candidate 0084's evaluate(State, perspective).
// The board uses the platform's signed piece encoding and contains 60 squares.
double heuristic_084_raw_value(const int* board, int perspective);

} // namespace junqi::az

#endif
