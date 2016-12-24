
include_directories(
  model
)

add_executable(UniGramLabeler UniGramLabeler.cpp)
target_link_libraries(UniGramLabeler ${LIBS})
