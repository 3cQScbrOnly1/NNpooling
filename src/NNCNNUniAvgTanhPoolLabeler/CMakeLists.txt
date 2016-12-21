
include_directories(
  model
)

add_executable(NNCNNUniAvgTanhPoolLabeler NNCNNUniAvgTanhPoolLabeler.cpp)
target_link_libraries(NNCNNUniAvgTanhPoolLabeler ${LIBS})
