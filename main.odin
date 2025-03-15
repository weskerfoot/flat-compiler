package main

import "core:fmt"
import "core:unicode"
import "core:strconv"
import "core:container/queue"
import "core:text/regex"

OpAssoc :: enum{NoAssoc, Left, Right}

OpIndex :: distinct int
InfixOperator :: struct {
  prec: int,
  assoc: OpAssoc
}

Token :: struct {
  token: string,
  type: TokenType,
  infix_op: Maybe(InfixOperator)
}

TokenType :: enum{Number, Ident, InfixOp, Paren, SemiColon}

// TODO add distinction between Variable, TypeName, and FunctionName instead of just Identifier
NodeType :: enum{Application, Identifier, Number, Root, InfixOp}

ValueType :: enum{Integer, String, Function}

ParseNode :: struct {
  tokenIndex: int,
  nodeType: NodeType
}

get_parsed_token_value :: proc(parseNode: ParseNode,
                               parseState: ParseState) -> Token {
  tokenIndex: int = parseNode.tokenIndex
  return parseState.tokens[tokenIndex]
}

ParserStates :: enum{App, NonTerminal, Terminal}

ParseState :: struct {
  nodeType: NodeType,
  tokenIndex: int,
  state: ParserStates,
  parsingInfix: bool,
  tokens: ^#soa[dynamic]Token,
  node_queue: ^queue.Queue(ParseNode),
  node_stack: ^queue.Queue(ParseNode),
}

// Used as SoA to store raw data
// TODO, you would map these to a scope as well
// the scopes could simply be indices that then map to another
// table where the actual code for a function is stored (and possibly parent scopes if I decide to add closures)
RawValues :: struct {
  float: ^[dynamic]f64,
  integer: ^[dynamic]int,
  str: ^[dynamic]string
}

ValueLookaside :: struct {
  valueIndex: int,
  valueType: ValueType
}

lookahead_basic :: proc(tokens: ^#soa[dynamic]Token,
                        start: int,
                        n: int) -> Maybe(Token) {
  if start+n >= len(tokens) {
    return nil
  }
  return tokens[start+n]
}

lookahead_with_tokvalue :: proc(tokens: ^#soa[dynamic]Token,
                                start: int,
                                n: int,
                                tokValue: string) -> Maybe(Token) {
  if start+n >= len(tokens) {
    return nil
  }
  result := tokens[start+n]
  if (result.token != tokValue) {
    return nil
  }
  return result
}

lookahead :: proc{lookahead_with_tokvalue, lookahead_basic}

check_operator :: proc(st: string) -> Maybe(int) {
  // Returns the number of tokens to seek ahead or nil
  switch st[0:2] {
    case ":=":
      return 2
  }
  switch rune(st[0]) {
    case '+', '-','*', '/', '^', ',':
      return 1
  }
  return nil
}

is_operator_char :: proc(c : rune) -> bool {
  return (c == '*' ||
          c == '/' ||
          c == '+' ||
          c == '-' ||
          c == ',' ||
          c == '=' ||
          c == ':' ||
          c == '^')
}

op_prec_assoc :: proc(op_st : string) -> (Maybe(int), Maybe(OpAssoc)) {
  result: int
  switch op_st {
    case ":=": // := has the lowest precedence so, e.g. a = 1 + 2 * 3, gets parsed into a 1 2 3 * + =
      return 1, OpAssoc.Right
    case "+", "-":
      return 2, OpAssoc.Left
    case "*", "/":
      return 3, OpAssoc.Left
    case "^":
      return 4, OpAssoc.Right
    case ",":
      return 5, OpAssoc.Left
  }
  return nil, nil
}

get_op :: proc(current_tok_start: int, input_st : string) -> (InfixOperator, int) {
  num_tokens, is_op := check_operator(input_st[current_tok_start:]).?

  assert(is_op, "Should always be a valid op if at least one op character was found")

  op_st := input_st[current_tok_start:(current_tok_start+num_tokens)]

  maybe_op_prec, maybe_op_assoc := op_prec_assoc(op_st)

  op_prec, op_prec_ok := maybe_op_prec.?
  op_assoc, op_assoc_ok := maybe_op_assoc.?

  assert (op_prec_ok && op_assoc_ok, "Should always be precedence and associativity for an operator")

  return InfixOperator{op_prec, op_assoc}, current_tok_start+num_tokens
}

is_identifier_char :: proc(c : rune) -> bool {
  return !is_operator_char(c) && !unicode.is_number(c) && !unicode.is_space(c) && c != '(' && c != ')' && c != ';'
}

is_whitespace :: proc(c: rune) -> bool {
  return unicode.is_space(c)
}

tokenize :: proc(input_st: string, tokens: ^#soa[dynamic]Token) {
  current_tok_start: int = 0
  current_tok_end: int = 1

  current_tok_type: TokenType

  for current_tok_start < len(input_st) {
    c := rune(input_st[current_tok_start])
    //fmt.println("current c: ", c)

    if c == '(' || c == ')' {
      current_tok_type = TokenType.Paren
      append(tokens, Token{input_st[current_tok_start:current_tok_end], current_tok_type, nil})
    }

    if c == ';' {
      current_tok_type = TokenType.SemiColon
      append(tokens, Token{input_st[current_tok_start:current_tok_end], current_tok_type, nil})
    }

    if unicode.is_number(c) {
      current_tok_type = TokenType.Number

      for current_tok_end < len(input_st) {
        next_digit := rune(input_st[current_tok_end])
        if current_tok_end > len(input_st) || !unicode.is_number(next_digit) {
          break
        }
        current_tok_end += 1
      }
      append(tokens, Token{input_st[current_tok_start:current_tok_end], current_tok_type, nil})
    }

    if is_operator_char(c) {
      current_tok_type = TokenType.InfixOp

      op_info, op_end := get_op(current_tok_start, input_st)
      op_st := input_st[current_tok_start:op_end]

      append(tokens, Token{op_st, current_tok_type, op_info})

      current_tok_end += (op_end - current_tok_end)
    }

    if is_identifier_char(c) {
      current_tok_type = TokenType.Ident
      for current_tok_end < len(input_st) {
        next_letter := rune(input_st[current_tok_end])

        if current_tok_end >= len(input_st) || !is_identifier_char(next_letter) {
          //fmt.println("breaking end of identifier with character ", next_letter)
          break
        }
        current_tok_end += 1
      }

      append(tokens, Token{input_st[current_tok_start:current_tok_end], current_tok_type, nil})
    }

    current_tok_start = current_tok_end
    current_tok_end += 1
  }
}

get_parse_node :: proc(parseState: ParseState, tokenOffset: int) -> ParseNode {
  return ParseNode{parseState.tokenIndex-tokenOffset, parseState.nodeType}
}

skip_tokens :: proc(parserState: ParseState,
                    nTokensAdvance: int) -> ParseState {
  newParserState: ParseState = parserState
  newParserState.tokenIndex += nTokensAdvance
  newParserState.nodeType = parserState.nodeType
  newParserState.node_queue = parserState.node_queue
  newParserState.node_stack = parserState.node_stack
  newParserState.tokens = parserState.tokens
  newParserState.state = parserState.state

  assert ((newParserState.tokenIndex)-nTokensAdvance <= len(newParserState.tokens))
  return newParserState
}

advance_parser :: proc(parserState: ParseState,
                       nodeType: NodeType,
                       nTokensAdvance: int,
                       newState: ParserStates) -> ParseState {
  newParserState: ParseState = parserState
  newParserState.tokenIndex += nTokensAdvance
  newParserState.nodeType = nodeType
  newParserState.node_queue = parserState.node_queue
  newParserState.node_stack = parserState.node_stack
  newParserState.tokens = parserState.tokens
  newParserState.state = newState

  queue.push_back(newParserState.node_queue, get_parse_node(newParserState, nTokensAdvance))

  assert ((newParserState.tokenIndex)-nTokensAdvance <= len(newParserState.tokens))
  return newParserState
}

drain_stack :: proc(parseState: ParseState) -> ParseState {
  for queue.len(parseState.node_stack^) > 0 {
    queue.push_front(parseState.node_queue, queue.pop_back(parseState.node_stack))
  }
  return parseState
}

reset_node_type :: proc(parserState: ParseState, nodeType: NodeType) -> ParseState {
  newParserState := parserState
  queue.peek_front(newParserState.node_queue).nodeType = nodeType
  return newParserState
}

get_latest_token :: proc(parserState: ParseState) -> Maybe(Token) {
  if (parserState.tokenIndex >= len(parserState.tokens)) {
    return nil
  }
  return parserState.tokens[parserState.tokenIndex]
}

expect_token_st :: #force_inline proc(parserState: ParseState,
                                      token_value: string,
                                      lineno: int) {
  if parserState.tokenIndex >= len(parserState.tokens) {
    return
  }
  // TODO, include the line in the source code itself
  if !(parserState.tokens[parserState.tokenIndex].token == token_value) {
    fmt.panicf("Expected \"%s\" but actual token is \"%s\", line in parser = %d\n",
               token_value,
               parserState.tokens[parserState.tokenIndex].token,
               lineno)
  }
}

expect_token_type :: #force_inline proc(parserState: ParseState,
                                        token_type: TokenType,
                                        lineno: int) {
  // TODO, include the line in the source code itself
  if parserState.tokenIndex >= len(parserState.tokens) {
    return
  }
  if !(parserState.tokens[parserState.tokenIndex].type == token_type) {
    fmt.panicf("Expected token of type \"%s\" but actual type is \"%s\", with value \"%s\", line in parser = %d\n",
               token_type,
               parserState.tokens[parserState.tokenIndex].type,
               parserState.tokens[parserState.tokenIndex].token,
               lineno)
  }
}

expect_not_token_type :: #force_inline proc(parserState: ParseState,
                                       token_type: TokenType,
                                       lineno: int) {
  if parserState.tokenIndex >= len(parserState.tokens) {
    return
  }
  if (parserState.tokens[parserState.tokenIndex].type == token_type) {
    fmt.panicf("Didn't expect token of type \"%s\" but it is \"%s\", line in parser = %d\n",
               token_type,
               parserState.tokens[parserState.tokenIndex].type,
               lineno)
  }
}

expect_not_token_st :: #force_inline proc(parserState: ParseState,
                                          token_value: string,
                                          lineno: int) {
  // TODO, include the line in the source code itself
  if parserState.tokenIndex >= len(parserState.tokens) {
    return
  }
  if parserState.tokens[parserState.tokenIndex].token == token_value {
    fmt.panicf("Didn't expect token \"%s\" but it is \"%s\", line in parser = %d\n",
               token_value,
               parserState.tokens[parserState.tokenIndex].token,
               lineno)
  }
}


expect_not_token :: proc{expect_not_token_st, expect_not_token_type}
expect_token :: proc{expect_token_st, expect_token_type}

get_node_index :: proc(parserState: ParseState) -> int {
  return queue.len(parserState.node_stack^)-1
}

parse_sep_by :: proc(parserState: ParseState, sep: string) -> ParseState {
  //fmt.println("Parsing sepby: sep =", sep)
  curParserState: ParseState = parserState

  assert (curParserState.tokenIndex < len(curParserState.tokens), "Should never seek beyond tokens length")

  for curParserState.tokenIndex < len(curParserState.tokens) {
    curParserState = parse(curParserState)

    if curParserState.tokenIndex == len(curParserState.tokens) {
      break
    }

    expect_token(curParserState, sep, #line)
    curParserState = skip_tokens(curParserState, 1)
    curParserState.parsingInfix = false
  }
  return curParserState
}

parse_infix :: proc(parserState: ParseState, minPrec: int) -> ParseState {
  if parserState.tokenIndex >= len(parserState.tokens) {
    return parserState
  }

  //fmt.println(parserState.tokens[parserState.tokenIndex:])
  assert (parserState.tokenIndex < len(parserState.tokens), "Should never seek beyond tokens length")
  curParserState: ParseState

  // If we just started parsing the infix expr, assume the lhs has already been parsed
  if parserState.tokenIndex < len(parserState.tokens) && parserState.tokens[parserState.tokenIndex].token == ")" {
    return curParserState
  }

  if !parserState.parsingInfix {
    curParserState = parserState
    expect_token(curParserState, TokenType.InfixOp, #line)
  }
  else {
    // otherwise parse it, and it will stop once it hits an infix op
    //fmt.println("parsing lhs")
    curParserState = parse(parserState)
  }
  curParserState.parsingInfix = true

  for true {
    if curParserState.tokenIndex >= len(curParserState.tokens) {
      //fmt.println("breaking because we consumed all tokens")
      break
    }

    cur_token: Token = curParserState.tokens[curParserState.tokenIndex]
    if cur_token.type != TokenType.InfixOp || cur_token.infix_op.?.prec < minPrec {
      //fmt.println("breaking because the either we encountered no infix op, or current operator has prec < minPrec, ", cur_token, minPrec)
      break
    }

    if cur_token.token == ")" {
      //fmt.println("found a right paren so breaking")
      curParserState = skip_tokens(curParserState, 1)
      curParserState.parsingInfix = false
      break
    }

    assert (cur_token.infix_op != nil, "Should always be a valid infix op here")

    prec := cur_token.infix_op.?.prec
    assoc := cur_token.infix_op.?.assoc

    //fmt.println(prec, assoc, cur_token.token)

    nextMinPrec: int
    if assoc == OpAssoc.Left {
      nextMinPrec = prec + 1
    }
    else {
      nextMinPrec = prec
    }

    curParserState.nodeType = NodeType.InfixOp
    infix_op := get_parse_node(curParserState, 0)

    curParserState.tokenIndex += 1

    assert (curParserState.tokens[curParserState.tokenIndex].type != TokenType.InfixOp, "Should never be an infix op here")

    curParserState = parse_infix(curParserState, nextMinPrec)
    expect_not_token(curParserState, "(", #line)

    queue.push_back(curParserState.node_queue, infix_op)

  }
  return curParserState
}

parse_application :: proc(parserState: ParseState) -> ParseState {
  //fmt.println("Parsing application")
  curParserState := parserState

  expect_token(curParserState, TokenType.Ident, #line)

  func_name := get_parse_node(curParserState, 0)
  //fmt.println(func_name)
  curParserState = skip_tokens(curParserState, 1)
  curParserState.parsingInfix = true // TODO check for comma separated list

  curParserState = parse(curParserState)
  queue.push_back(curParserState.node_queue, func_name)

  // If this is not an infix expression it will just continue parsing normally
  return parse_infix(curParserState, 1)
}

parse :: proc(parserState: ParseState) -> ParseState {

  if parserState.tokenIndex >= len(parserState.tokens) {
    //fmt.println("we've consumed all tokens")
    return parserState
  }

  newParserState: ParseState = parserState

  token: Token = parserState.tokens[parserState.tokenIndex]

  switch token.type {
    case TokenType.Number:
      //fmt.println("number")
      newParserState = advance_parser(parserState, NodeType.Number, 1, ParserStates.Terminal)
      return parse(newParserState)
    case TokenType.InfixOp:
      if parserState.parsingInfix {
        // If we are already parsing an infix expression and this is an infix operator, simply return
        // the parse_infix function will handle consuming the next token
        //fmt.println("in parse and there was an infix op and we were already parsing infix")
        return parserState
      }
      else {
        // The identifier represents an infix operator, so this is an infix expression
        // and since we're not already parsing an infix expression, kick off the infix parser
        //fmt.println("infix application")
        newParserState = parse_infix(parserState, 1)
        return newParserState
      }
    case TokenType.Ident:
      //fmt.println("identifier")

      // Check if it's a left paren, then it's a function application
      // Need to check the *next token*
      left_paren, tokens_ok := lookahead(newParserState.tokens, newParserState.tokenIndex, 1).?

      //fmt.println(left_paren)

      if tokens_ok && left_paren.token == "(" {
        //fmt.println("application")
        newParserState = parse_application(newParserState)
        //fmt.println("done parsing application")
      }
      else {
        newParserState = advance_parser(parserState, NodeType.Identifier, 1, ParserStates.Terminal)
      }

      return parse(newParserState)
    case TokenType.Paren:
      if token.token == ")" {
        return parserState
      }
      else if parserState.parsingInfix {
        // TODO, test to see what happens if this isn't actually an infix expression
        // what should it do in that case? let parse_infix parse the lhs then just return?
        // that should work reliably
        //fmt.println("Starting a new parenthesized infix parse")
        expect_token(parserState, "(", #line)
        newParserState = parserState
        newParserState = skip_tokens(parserState, 1)
        newParserState.parsingInfix = true
        newParserState = parse_infix(newParserState, 1)
        expect_token(newParserState, ")", #line)
        newParserState = skip_tokens(newParserState, 1)
        return newParserState
      }
      else {
        newParserState = parserState
        newParserState.parsingInfix = true
        return parse_infix(newParserState, 1)
      }
    case TokenType.SemiColon:
      return parserState
  }
  return newParserState
}

print_tokens_as_rpn :: proc(node_queue: ^queue.Queue(ParseNode),
                            parseState: ParseState) {
  for queue.len(node_queue^) > 0 {
    node := queue.pop_front(node_queue)
    fmt.printf("%s ", parseState.tokens[node.tokenIndex].token)
  }
  fmt.println("")
}

get_value :: proc(runtime_index: int,
                  runtime_data: ^#soa[dynamic]ValueLookaside,
                  raw_values: ^[dynamic]$T) -> T {
  runtime_value := runtime_data[runtime_index]
  value_index := runtime_value.valueIndex
  return raw_values[value_index]
}

interp :: proc(node_queue: ^queue.Queue(ParseNode),
               parseState: ParseState,
               evaluation_stack: ^queue.Queue(int),
               runtime_data: ^#soa[dynamic]ValueLookaside,
               raw_values: RawValues) -> ^queue.Queue(int) {

  // ValueType :: enum{Integer, String, Function}

  for queue.len(node_queue^) > 0 {
    parseNode := queue.pop_front(node_queue)
    switch parseNode.nodeType {
      case NodeType.Application:
        fmt.println("application")
      case NodeType.Identifier:
        fmt.println("identifier")
      case NodeType.Number:
        fmt.println("number")

        // TODO, add a function that handles the appends and push and stuff
        // make it polymorphic so I can re-use it
        tok: Token = get_parsed_token_value(parseNode, parseState)
        value, ok := strconv.parse_int(tok.token) // TODO tokenize floats, uints, etc
        append(raw_values.integer, value)
        append(runtime_data, ValueLookaside{len(raw_values.integer)-1, ValueType.Integer})
        queue.push_front(evaluation_stack, len(runtime_data)-1)

      case NodeType.Root:
        fmt.println("root")
      case NodeType.InfixOp:
        fmt.println("infix op")
        tok: Token = get_parsed_token_value(parseNode, parseState)
        left := queue.pop_front(evaluation_stack)
        right := queue.pop_front(evaluation_stack)

        // TODO, you would have to dispatch on the type instead of always using integers
        left_val := get_value(left, runtime_data, raw_values.integer)
        right_val := get_value(right, runtime_data, raw_values.integer)
        switch tok.token {
          case "*":
            fmt.println(left_val, "*", right_val)
            append(raw_values.integer, left_val * right_val)
          case "+":
            fmt.println(left_val, "+", right_val)
            append(raw_values.integer, left_val + right_val)
          case "-":
            fmt.println(right_val, "-", left_val)
            append(raw_values.integer, right_val - left_val)
          case "/":
            fmt.println(right_val, "/", left_val)
            append(raw_values.integer, right_val / left_val)
        }
        append(runtime_data, ValueLookaside{len(raw_values.integer)-1, ValueType.Integer})
        queue.push_front(evaluation_stack, len(runtime_data)-1)
    }
  }
  return evaluation_stack
}

main :: proc() {
  //test_string: string = "foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  //test_string: string = "1 + 111 / (2 - (4 +5)) *(99/ 4)"
  //test_string: string = "a := 1 + 23 + 2 * 3"
  //test_string: string = "1 * 2 + 12 * cos((3 / 4) - 14)"
  //test_string: string = "cos(12 + 4) a(1,2)"
  //test_string: string = "foobar := sin(14 + 12) * cos(2 - 3); a + b * c"
  test_string: string = "2 + ((3 * 4) + 7)"
  tokens: #soa[dynamic]Token
  node_stack: queue.Queue(ParseNode)
  node_queue: queue.Queue(ParseNode)

  runtime_data: #soa[dynamic]ValueLookaside
  evaluation_stack: queue.Queue(int)
  raw_values: RawValues
  float_values: [dynamic]f64
  string_values: [dynamic]string // TODO optimize repeated strings?
  int_values: [dynamic]int
  raw_values.float = &float_values
  raw_values.integer = &int_values
  raw_values.str = &string_values

  tokenize(test_string, &tokens)

  parseState := ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, &tokens, &node_queue, &node_stack}
  parseState = parse_sep_by(parseState, ";")

  //print_tokens_as_rpn(&node_queue, parseState)
  fmt.println(interp(&node_queue, parseState, &evaluation_stack, &runtime_data, raw_values))
  //fmt.println(runtime_data)
  //fmt.println(raw_values)
  //fmt.println(tokens)
}
