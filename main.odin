package main

import "core:os"
import "core:fmt"
import "core:unicode"
import "core:strconv"
import "core:container/queue"

OpAssoc :: enum{NoAssoc, Left, Right}

OpIndex :: distinct int
InfixOperator :: struct {
  prec: int,
  assoc: OpAssoc,
  unary: int
}

Token :: struct {
  token: string,
  type: TokenType,
  infix_op: Maybe(InfixOperator)
}

TokenType :: enum{Number, Ident, InfixOp, Paren, SemiColon}

// TODO add distinction between Variable, TypeName, and FunctionName instead of just Identifier
NodeType :: enum{Application, Identifier, Number, Root, InfixOp, UnaryOp}

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
  parsingUnary: bool,
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

get_op_info :: proc(op_st : string) -> (Maybe(int), Maybe(OpAssoc), Maybe(int)) {
  switch op_st {
    case ":=": // := has the lowest precedence so, e.g. a = 1 + 2 * 3, gets parsed into a 1 2 3 * + =
      return 1, OpAssoc.Right, 0
    case "+", "-":
      return 2, OpAssoc.Left, 1 // 1 = it's able to be used as a unary op
    case "*", "/":
      return 3, OpAssoc.Left, 0
    case "^":
      return 4, OpAssoc.Right, 0
    case ",":
      return 5, OpAssoc.Left, 0
  }
  return nil, nil, nil
}

get_op :: proc(current_tok_start: int, input_st : string) -> (InfixOperator, int) {
  num_tokens, is_op := check_operator(input_st[current_tok_start:]).?

  assert(is_op, "Should always be a valid op if at least one op character was found")

  op_st := input_st[current_tok_start:(current_tok_start+num_tokens)]

  maybe_op_prec, maybe_op_assoc, maybe_is_unary := get_op_info(op_st)

  op_prec, op_prec_ok := maybe_op_prec.?
  op_assoc, op_assoc_ok := maybe_op_assoc.?
  is_unary, _ := maybe_is_unary.?

  assert (op_prec_ok && op_assoc_ok, "Should always be precedence and associativity for an operator")

  return InfixOperator{op_prec, op_assoc, is_unary}, current_tok_start+num_tokens
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
  // this algorithm is based on the "precedence climbing" algorithm
  // see https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing
  // it is tweaked to work without requiring an AST and also allows for unary operators
  if parserState.tokenIndex >= len(parserState.tokens) {
    return parserState
  }

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
    curParserState = parse(parserState)
  }
  curParserState.parsingInfix = true

  for true {
    if curParserState.tokenIndex >= len(curParserState.tokens) {
      break
    }

    cur_token: Token = curParserState.tokens[curParserState.tokenIndex]
    if cur_token.type != TokenType.InfixOp || cur_token.infix_op.?.prec < minPrec {
      break
    }

    if cur_token.token == ")" {
      curParserState = skip_tokens(curParserState, 1)
      curParserState.parsingInfix = false
      break
    }

    assert (cur_token.infix_op != nil, "Should always be a valid infix op here")

    prec := cur_token.infix_op.?.prec
    assoc := cur_token.infix_op.?.assoc

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

    if len(curParserState.tokens) <= curParserState.tokenIndex {
      fmt.panicf("Reached end of expression while expecting more tokens")
    }

    // Should never be an infix op *unless it's a unary op*
    // then we will parse that out in the main parser function as a unary expression
    // then when it comes back to this function it should be back parsing normal infix expressions
    if curParserState.tokens[curParserState.tokenIndex].type == TokenType.InfixOp {
      check_infix_op, infix_op_ok := curParserState.tokens[curParserState.tokenIndex].infix_op.?
      assert (infix_op_ok, "must be an infix operator here")
      assert (check_infix_op.unary == 1, "must be a unary operator if it's an operator here")
      curParserState.parsingUnary = true
      curParserState = parse(curParserState)
    }

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
  curParserState = skip_tokens(curParserState, 1)
  curParserState.parsingInfix = true // TODO check for comma separated list

  curParserState = parse(curParserState)
  queue.push_back(curParserState.node_queue, func_name)

  // If this is not an infix expression it will just continue parsing normally
  result := parse_infix(curParserState, 1)
  expect_not_token(curParserState, TokenType.Number, #line)
  expect_not_token(curParserState, TokenType.Ident, #line)
  return result
}

parse_unary :: proc(parserState: ParseState) -> ParseState {
  curParserState := parserState
  check_infix_op, infix_op_ok := curParserState.tokens[curParserState.tokenIndex].infix_op.?
  if !infix_op_ok {
    fmt.panicf("Expected an operator token when parsing unary expression")
  }
  assert (check_infix_op.unary == 1, "must be a unary operator")

  // Need to set this so get_parse_node has the correct type
  curParserState.nodeType = NodeType.UnaryOp
  unary_op := get_parse_node(curParserState, 0)

  curParserState = skip_tokens(curParserState, 1)

  // No longer parsing a unary op
  curParserState.parsingUnary = false
  curParserState = parse(curParserState)

  queue.push_back(curParserState.node_queue, unary_op)

  return curParserState
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
      fmt.println("number")
      newParserState = advance_parser(parserState, NodeType.Number, 1, ParserStates.Terminal)
      expect_not_token(newParserState, TokenType.Number, #line)
      expect_not_token(newParserState, TokenType.Ident, #line)
      return parse(newParserState)
    case TokenType.InfixOp:
      check_infix_op, infix_op_ok := newParserState.tokens[newParserState.tokenIndex].infix_op.?
      if !infix_op_ok {
        fmt.panicf("Encountered an invalid infix operator")
      }
      if check_infix_op.unary == 1 && parserState.parsingUnary {
        fmt.println("got a unary op in parse")
        newParserState = parse_unary(newParserState)
        return newParserState
      }
      if parserState.parsingInfix {
        // If we are already parsing an infix expression and this is an infix operator, simply return
        // the parse_infix function will handle consuming the next token
        //fmt.println("in parse and there was an infix op and we were already parsing infix")
        // unless it's a unary operator then you want to parse that
        return parserState
      }
      else {
        // The identifier represents an infix operator, so this is an infix expression
        // and since we're not already parsing an infix expression, kick off the infix parser
        fmt.println("starting new infix parse")
        newParserState = parse_infix(parserState, 1)
        return newParserState
      }
    case TokenType.Ident:
      // Check if it's a left paren, then it's a function application
      // Need to check the *next token*
      left_paren, tokens_ok := lookahead(newParserState.tokens, newParserState.tokenIndex, 1).?

      if tokens_ok && left_paren.token == "(" {
        newParserState = parse_application(newParserState)
      }
      else {
        newParserState = advance_parser(parserState, NodeType.Identifier, 1, ParserStates.Terminal)
      }

      expect_not_token(newParserState, TokenType.Ident, #line)
      expect_not_token(newParserState, TokenType.Number, #line)

      return parse(newParserState)
    case TokenType.Paren:
      if token.token == ")" {
        return parserState
      }
      else if parserState.parsingInfix {
        // TODO, test to see what happens if this isn't actually an infix expression
        // what should it do in that case? let parse_infix parse the lhs then just return?
        // that should work reliably
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
    node, ok := queue.pop_front_safe(node_queue)
    if ok {
      fmt.printf("%s ", parseState.tokens[node.tokenIndex].token)
    }
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

  fmt.println("interpreting")
  // ValueType :: enum{Integer, String, Function}

  for queue.len(node_queue^) > 0 {
    parseNode, parse_ok := queue.pop_front_safe(node_queue)
    fmt.println(parseNode)
    if !parse_ok {
      fmt.println("node queue empty, finishing")
      return evaluation_stack
    }
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
        value, num_ok := strconv.parse_int(tok.token) // TODO tokenize floats, uints, etc
        if !num_ok {
          fmt.panicf("Encountered an invalid integer value \"%s\"", tok.token)
        }
        append(raw_values.integer, value)
        append(runtime_data, ValueLookaside{len(raw_values.integer)-1, ValueType.Integer})
        queue.push_front(evaluation_stack, len(runtime_data)-1)

      case NodeType.Root:
        fmt.println("root")
      case NodeType.UnaryOp:
        fmt.println("unary op")
        tok: Token = get_parsed_token_value(parseNode, parseState)
        val_node, val_node_ok := queue.pop_front_safe(evaluation_stack)

        if !val_node_ok {
          fmt.panicf("Encountered an invalid value node")
        }

        val := get_value(val_node, runtime_data, raw_values.integer)
        fmt.println(val)
        switch tok.token {
          case "+":
            fmt.println("+", val)
            append(raw_values.integer, +val)
          case "-":
            fmt.println("-", val)
            append(raw_values.integer, -val)
        }
        append(runtime_data, ValueLookaside{len(raw_values.integer)-1, ValueType.Integer})
        queue.push_front(evaluation_stack, len(runtime_data)-1)

      case NodeType.InfixOp:
        fmt.println("infix op")
        tok: Token = get_parsed_token_value(parseNode, parseState)
        left, left_ok := queue.pop_front_safe(evaluation_stack)
        right, right_ok := queue.pop_front_safe(evaluation_stack)

        if !(left_ok && right_ok) {
          fmt.println("reached end of tokens to evaluate")
          return evaluation_stack
        }
        if !right_ok {
          fmt.println("Error, missing right operand")
        }

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
  fmt.println("top value = ", queue.peek_front(evaluation_stack))
  return evaluation_stack
}

main :: proc() {
  input: [256]byte
  n_chars, err := os.read(os.stdin, input[:])

  if n_chars <= 0 {
    fmt.panicf("Failed to read input, err = %s", err)
  }

  test_string: string = string(input[:n_chars])
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

  parseState := ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, false, &tokens, &node_queue, &node_stack}
  parseState = parse_sep_by(parseState, ";")

  print_tokens_as_rpn(&node_queue, parseState)
}
