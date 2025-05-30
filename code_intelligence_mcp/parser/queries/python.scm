; Tree-sitter query for Python symbols

; Function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body) @function.definition

; Async function definitions
(function_definition
  (async) @function.async
  name: (identifier) @function.name) @async_function

; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.definition

; Method definitions (functions inside classes)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.params) @method.definition))

; Decorators
(decorated_definition
  (decorator) @decorator
  definition: (_) @decorated)

; Import statements
(import_statement
  name: (dotted_name) @import.module) @import

(import_from_statement
  module_name: (dotted_name) @import.module
  name: (_) @import.name) @import_from

; Function calls
(call
  function: (identifier) @call.function
  arguments: (argument_list) @call.args) @call

; Method calls
(call
  function: (attribute
    object: (_) @call.object
    attribute: (identifier) @call.method)) @method_call

; Assignments
(assignment
  left: (identifier) @assignment.target
  right: (_) @assignment.value) @assignment

; Global variables
(module
  (expression_statement
    (assignment
      left: (identifier) @global.variable)))

; Docstrings
(function_definition
  body: (block
    (expression_statement
      (string) @docstring.content) @docstring)) 

(class_definition
  body: (block
    (expression_statement
      (string) @docstring.content) @docstring))

; Type annotations
(function_definition
  return_type: (type) @function.return_type)

(parameters
  (typed_parameter
    name: (identifier) @param.name
    type: (type) @param.type))

; Exception handling
(try_statement
  body: (block) @try.body
  (except_clause
    (as_pattern
      alias: (identifier) @except.name)) @except)

; List/dict comprehensions
(list_comprehension) @comprehension
(dictionary_comprehension) @comprehension
(set_comprehension) @comprehension

; Lambda functions
(lambda
  parameters: (lambda_parameters) @lambda.params
  body: (_) @lambda.body) @lambda

; Yield statements (generators)
(yield) @generator.yield
(yield_from) @generator.yield_from