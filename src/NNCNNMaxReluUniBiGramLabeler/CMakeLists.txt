
include_directories(
  model
)

add_executable(NNCNNMaxReluUniBiGramLabeler NNCNNMaxReluUniBiGramLabeler.cpp)
target_link_libraries(NNCNNMaxReluUniBiGramLabeler ${LIBS})
