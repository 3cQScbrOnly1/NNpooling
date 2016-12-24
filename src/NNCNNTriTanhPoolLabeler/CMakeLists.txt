
include_directories(
  model
)

add_executable(NNCNNTriTanhPoolLabeler NNCNNTriTanhPoolLabeler.cpp)
target_link_libraries(NNCNNTriTanhPoolLabeler ${LIBS})
