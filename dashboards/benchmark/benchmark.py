# dashboards/benchmark/benchmark.py
import os
from benchmark_layout import render_benchmark_dashboard

if __name__ == "__main__":
    import streamlit as st
    render_benchmark_dashboard()
