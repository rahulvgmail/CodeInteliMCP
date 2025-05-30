; Tree-sitter query for JavaScript symbols

; Function declarations
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  body: (statement_block) @function.body) @function.declaration

; Arrow functions
(arrow_function
  parameters: (_) @arrow.params
  body: (_) @arrow.body) @arrow.function

; Function expressions
(function_expression
  name: (identifier)? @function.name
  parameters: (formal_parameters) @function.params
  body: (statement_block) @function.body) @function.expression

; Variable declarations with functions
(variable_declarator
  name: (identifier) @variable.name
  value: (function_expression) @variable.function)

(variable_declarator
  name: (identifier) @variable.name
  value: (arrow_function) @variable.arrow)

; Class declarations
(class_declaration
  name: (identifier) @class.name
  heritage: (class_heritage)? @class.extends
  body: (class_body) @class.body) @class.declaration

; Method definitions
(method_definition
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  body: (statement_block) @method.body) @method.definition

; Constructor
(method_definition
  name: (identifier) @constructor.name (#eq? @constructor.name "constructor")
  parameters: (formal_parameters) @constructor.params
  body: (statement_block) @constructor.body) @constructor

; Import statements
(import_statement
  source: (string) @import.source) @import

(import_specifier
  name: (identifier) @import.name)

(namespace_import
  (identifier) @import.namespace)

; Export statements
(export_statement
  declaration: (_) @export.declaration) @export

(export_specifier
  name: (identifier) @export.name)

; Function calls
(call_expression
  function: (identifier) @call.function
  arguments: (arguments) @call.args) @call

; Method calls
(call_expression
  function: (member_expression
    object: (_) @call.object
    property: (property_identifier) @call.method)) @method_call

; Async functions
(function_declaration
  (async) @function.async
  name: (identifier) @function.name) @async_function

(arrow_function
  (async) @arrow.async) @async_arrow

; Generator functions
(function_declaration
  (generator_function) @generator
  name: (identifier) @generator.name) @generator_function

; Object properties
(property
  key: (property_identifier) @property.key
  value: (_) @property.value) @property

; Variable declarations
(variable_declaration
  (variable_declarator
    name: (identifier) @variable.name
    value: (_)? @variable.value)) @variable

; JSX elements
(jsx_element
  open_tag: (jsx_opening_element
    name: (identifier) @jsx.tag)) @jsx

(jsx_self_closing_element
  name: (identifier) @jsx.tag) @jsx.self_closing

; Try-catch blocks
(try_statement
  body: (statement_block) @try.body
  handler: (catch_clause
    parameter: (identifier)? @catch.param
    body: (statement_block) @catch.body)?) @try_catch

; Switch statements
(switch_statement
  value: (_) @switch.value
  body: (switch_body) @switch.body) @switch

; Object destructuring
(object_pattern
  (shorthand_property_identifier_pattern) @destructure.property)

; Array destructuring  
(array_pattern
  (identifier) @destructure.element)