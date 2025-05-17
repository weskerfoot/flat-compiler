package main

import "core:testing"
import "core:fmt"
import "core:log"
import "core:container/queue"

example_strings := []string{
  "1 + 111 / (2 - (4 +5)) *(99/ 4)",
  "2 + 3 * -12 / 2",
  "2 + 3 * 12 - 2",
  "10 + 20 * 3",
  "100 - 50 / 5",
  "8 * 3 + 6 / 2",
  "(4 + 6) * (3 - 1)",
  "50 / (2 + 3) + 4 * 2",
  "100 / 10 + 100 / 25",
  "7 * (8 - 3) - 10",
  "1 + 2 + 3 + 4 * 5",
  "100 - (20 + 5) * 2",
  "60 / 2 / 3",
  "81 / (3 * 3)",
  "5 + ((2 + 3) * (4 - 1))",
  "3+4*2-6/3",
  "18/(3+3)*2+1",
  "7*2+5 -3*1",
  "42-(6+8)/2*3",
  "(10+2)*(6-4)/2",
  "15+5*3-9/3",
  "6*3+12/(2+2)",
  "100-(8*4+4)/2",
  "2+2*2+2*2",
  "9*(2+1)-4*2",
  "50/5+6*2-3",
  "4+18/(3*2)-1",
  "25-(5+5)*2+10",
  "8*2-(6/3+1)",
  "(7+1)*(2+2)-10"
}

expected_outputs := []int{
  1 + 111 / (2 - (4 +5)) *(99/ 4),
  2 + 3 * -12 / 2,
  2 + 3 * 12 - 2,
  10 + 20 * 3,
  100 - 50 / 5,
  8 * 3 + 6 / 2,
  (4 + 6) * (3 - 1),
  50 / (2 + 3) + 4 * 2,
  100 / 10 + 100 / 25,
  7 * (8 - 3) - 10,
  1 + 2 + 3 + 4 * 5,
  100 - (20 + 5) * 2,
  60 / 2 / 3,
  81 / (3 * 3),
  5 + ((2 + 3) * (4 - 1)),
  3 + 4 * 2 - 6 / 3,
  18 / (3 + 3) * 2 + 1,
  7 * 2 + 5 - 3 * 1,
  42 - (6 + 8) / 2 * 3,
  (10 + 2) * (6 - 4) / 2,
  15 + 5 * 3 - 9 / 3,
  6 * 3 + 12 / (2 + 2),
  100 - (8 * 4 + 4) / 2,
  2 + 2 * 2 + 2 * 2,
  9 * (2 + 1) - 4 * 2,
  50 / 5 + 6 * 2 - 3,
  4 + 18 / (3 * 2) - 1,
  25 - (5 + 5) * 2 + 10,
  8 * 2 - (6 / 3 + 1),
  (7 + 1) * (2 + 2) - 10
}

test_examples := soa_zip(st_input=example_strings, expected_output=expected_outputs)

@(test)
test_parsing :: proc(t: ^testing.T) {
  for test_data, i in test_examples {
    st_input := test_data.st_input
    expected_output := test_data.expected_output

    tokens: #soa[dynamic]Token
    node_stack: queue.Queue(ParseNode)
    node_queue: queue.Queue(ParseNode)

    runtime_data: #soa[dynamic]ValueLookaside
    evaluation_stack: queue.Queue(int)

    // Set up arrays for raw values
    raw_values: RawValues
    float_values: [dynamic]f64
    string_values: [dynamic]string
    int_values: [dynamic]int
    raw_values.float = &float_values
    raw_values.integer = &int_values
    raw_values.str = &string_values

    // tokenize and parse input
    tokenize(st_input, &tokens)
    parseState := ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, false, &tokens, &node_queue, &node_stack}
    parseState = parse_sep_by(parseState, ";")

    // interpret the parsed input
    interp(&node_queue, parseState, &evaluation_stack, &runtime_data, raw_values)

    // get the final value on the stack
    actual_result := raw_values.integer[len(raw_values.integer)-1]
    error_string := fmt.aprintf("test number = %d, expected_result = %d, actual_result = %d",
                                i,
                                expected_output,
                                actual_result)

    testing.expect(t, actual_result == expected_output, error_string)
    log.info(error_string)
  }
}
