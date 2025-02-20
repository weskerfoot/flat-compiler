package main

import "core:fmt"
import "core:unicode"
import "core:strconv"
import "core:container/queue"

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

TokenType :: enum{Number, Ident, InfixOp, Paren}

// TODO add distinction between Variable, TypeName, and FunctionName instead of just Identifier
NodeType :: enum{Application, Identifier, Number, Root, InfixOp}

ParseNode :: struct {
  tokenIndex: int,
  nodeType: NodeType
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

is_operator_char :: proc(c : rune) -> bool {
  return (c == '*' ||
          c == '/' ||
          c == '+' ||
          c == '-' ||
          c == ',' ||
          c == '^')
}

op_prec :: proc(op_st : string) -> int {
  result: int
  switch op_st {
    case "+", "-":
      result = 1
    case "*", "/":
      result = 2
    case "^":
      result = 3
    case ",":
      result = 4
  }
  return result
}

op_assoc :: proc(op_st : string) -> OpAssoc {
  result: OpAssoc
  switch op_st {
    case "*", "/", "+", "-", ",":
      result = OpAssoc.Left
    case "^":
      result = OpAssoc.Right
  }
  return result
}

get_op :: proc(op_st : string) -> Maybe(InfixOperator) {
  if !is_operator_char(cast(rune)op_st[0]) {
    return nil
  }
  return InfixOperator{op_prec(op_st), op_assoc(op_st)}
}

is_identifier_char :: proc(c : rune) -> bool {
  return !is_operator_char(c) && !unicode.is_number(c) && !unicode.is_space(c) && c != '(' && c != ')'
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

    if c == '(' || c == ')' {
      current_tok_type = TokenType.Paren
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
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
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
    }

    if is_operator_char(c) {
      current_tok_type = TokenType.InfixOp

      for current_tok_end < len(input_st) {
        next_char := rune(input_st[current_tok_end])
        if current_tok_end > len(input_st) || !is_operator_char(next_char) {
          break
        }
        current_tok_end += 1
      }
      op_tok_str := string(input_st[current_tok_start:current_tok_end])
      op_info := get_op(op_tok_str)
      append(tokens, Token{op_tok_str, current_tok_type, op_info})
    }

    if is_identifier_char(c) {
      current_tok_type = TokenType.Ident
      for current_tok_end < len(input_st) {
        next_letter := rune(input_st[current_tok_end])

        if current_tok_end >= len(input_st) || !is_identifier_char(next_letter) {
          fmt.println("breaking end of identifier with character ", next_letter)
          break
        }
        current_tok_end += 1
      }

      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
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

parse_sep_by :: proc(parserState: ParseState, sep: string, end: string) -> ParseState {
  fmt.println("Parsing sepby: sep =", sep, ", end =", end)
  curParserState: ParseState = parserState

  assert (curParserState.tokenIndex < len(curParserState.tokens))

  for (curParserState.tokenIndex < len(curParserState.tokens) &&
       curParserState.tokens[curParserState.tokenIndex].token != end) {
    curParserState = parse(curParserState)

    if curParserState.tokenIndex >= len(curParserState.tokens) {
      fmt.panicf("Unexpected end of source, are you missing a \"%s\" somewhere?", end)
    }

    if curParserState.tokens[curParserState.tokenIndex].token == end {
      break
    }

    expect_token(curParserState, sep, #line)
    curParserState.tokenIndex += 1
  }
  return curParserState
}

parse_infix :: proc(parserState: ParseState, minPrec: int) -> ParseState {
  if parserState.tokenIndex >= len(parserState.tokens) {
    return parserState
  }

  fmt.println(parserState.tokens[parserState.tokenIndex:])
  assert (parserState.tokenIndex < len(parserState.tokens))
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
    fmt.println("parsing lhs")
    curParserState = parse(parserState)
  }
  curParserState.parsingInfix = true

  for true {
    if curParserState.tokenIndex >= len(curParserState.tokens) {
      fmt.println("breaking because we consumed all tokens")
      break
    }

    cur_token: Token = curParserState.tokens[curParserState.tokenIndex]
    if cur_token.type != TokenType.InfixOp || cur_token.infix_op.?.prec < minPrec {
      fmt.println("breaking because the either we encountered no infix op, or current operator has prec < minPrec, ", cur_token, minPrec)
      break
    }

    if cur_token.token == ")" {
      fmt.println("found a right paren so breaking")
      curParserState = skip_tokens(curParserState, 1)
      curParserState.parsingInfix = false
      break
    }

    assert (cur_token.infix_op != nil)

    prec := cur_token.infix_op.?.prec
    assoc := cur_token.infix_op.?.assoc

    fmt.println(prec, assoc, cur_token.token)

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

    assert (curParserState.tokens[curParserState.tokenIndex].type != TokenType.InfixOp)

    curParserState = parse_infix(curParserState, nextMinPrec)
    expect_not_token(curParserState, "(", #line)

    queue.push_back(curParserState.node_queue, infix_op)

  }
  return curParserState
}

parse_application :: proc(parserState: ParseState) -> ParseState {
  fmt.println("Parsing application")
  curParserState := parserState

  expect_token(curParserState, TokenType.Ident, #line)

  func_name := get_parse_node(curParserState, 0)
  fmt.println(func_name)
  curParserState = skip_tokens(curParserState, 1)
  curParserState.parsingInfix = true // TODO check for comma separated list

  curParserState = parse(curParserState)
  queue.push_back(curParserState.node_queue, func_name)

  // If this is not an infix expression it will just continue parsing normally
  return parse_infix(curParserState, 1)
}

parse :: proc(parserState: ParseState) -> ParseState {

  if parserState.tokenIndex >= len(parserState.tokens) {
    fmt.println("we've consumed all tokens")
    return parserState
  }

  newParserState: ParseState = parserState

  token: Token = parserState.tokens[parserState.tokenIndex]

  switch token.type {
    case TokenType.Number:
      fmt.println("number")
      newParserState = advance_parser(parserState, NodeType.Number, 1, ParserStates.Terminal)
      return parse(newParserState)
    case TokenType.InfixOp:
      if parserState.parsingInfix {
        // If we are already parsing an infix expression and this is an infix operator, simply return
        // the parse_infix function will handle consuming the next token
        fmt.println("in parse and there was an infix op and we were already parsing infix")
        return parserState
      }
      else {
        // The identifier represents an infix operator, so this is an infix expression
        // and since we're not already parsing an infix expression, kick off the infix parser
        fmt.println("infix application")
        newParserState = parse_infix(parserState, 1)
        return newParserState
      }
    case TokenType.Ident:
      fmt.println("identifier")

      // Check if it's a left paren, then it's a function application
      // Need to check the *next token*
      left_paren, tokens_ok := lookahead(newParserState.tokens, newParserState.tokenIndex, 1).?

      fmt.println(left_paren)

      if tokens_ok && left_paren.token == "(" {
        fmt.println("application")
        newParserState = parse_application(newParserState)
        fmt.println("done parsing application")
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
        fmt.println("Starting a new parenthesized infix parse")
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

main :: proc() {
  //test_string: string = "foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  test_string: string = "1 + 111 / (2 - (4 + 5)) * (99 / 4)"
  //test_string: string = "1 * 2 + 12 * cos((3 / 4) - 14)"
  //test_string: string = "cos(12 + 4) a(1,2)"
  //test_string: string = "sin(14 + 12) * cos(2 - 3)"
  tokens: #soa[dynamic]Token
  node_stack: queue.Queue(ParseNode)
  node_queue: queue.Queue(ParseNode)

  tokenize(test_string, &tokens)

  fmt.println("tokens at beginning")
  for tok in tokens {
    fmt.println(tok)
  }
  fmt.println("=============")

  parseState := parse(ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, &tokens, &node_queue, &node_stack})

  fmt.println(test_string)
  print_tokens_as_rpn(&node_queue, parseState)
}
