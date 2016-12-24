
include_directories(
  model
)

add_executable(UniBiTriGramLabeler UniBiTriGramLabeler.cpp)
target_link_libraries(UniBiTriGramLabeler ${LIBS})
