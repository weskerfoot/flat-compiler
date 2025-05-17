package main

import "core:testing"
import "core:fmt"
import "core:log"
import "core:container/queue"

@(test)
test_parsing :: proc(t: ^testing.T) {
  test_string: string = "2 + 3 * 12 - 2"
  expected_result : int = 2 + 3 * 12 - 2

  // queues / array for parsing
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
  tokenize(test_string, &tokens)
  parseState := ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, false, &tokens, &node_queue, &node_stack}
  parseState = parse_sep_by(parseState, ";")

  // interpret the parsed input
  interp(&node_queue, parseState, &evaluation_stack, &runtime_data, raw_values)

  // get the final value on the stack
  actual_result := raw_values.integer[len(raw_values.integer)-1]
  error_string := fmt.aprintf("expected_result = %d, actual_result = %d",
                             expected_result,
                             actual_result)

  testing.expect(t, actual_result == expected_result, error_string)
  log.info(error_string)
}
