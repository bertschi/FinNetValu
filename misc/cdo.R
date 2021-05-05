## Visualize Julia data

library(tidyverse)
library(ggthemes)

df <- read_csv("/tmp/foo.csv")

df %>%
    group_by(equity, eqtype, debt, dbtype,
             alpha_beta, r, tau, sigma, a0, N) %>%
    mutate(data = loss,
           dbfrac = 100 * (1 - 1 / (1 + debt))) %>%
    arrange(data) %>%
    mutate(q = 1:n() / n()) %>%
    ungroup() %>%
    ggplot(aes(q, data,
               linetype = dbtype,
               color = interaction(100 * equity, eqtype))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(alpha_beta ~ dbfrac,
               scales = "free") +
    theme_tufte() +
    labs(x = "Quantile",
         y = "Loss",
         color = "Equity fraction and type",
         linetype = "Debt type")

ggsave("/tmp/rand_loss.pdf")

df %>%
    group_by(equity, eqtype, debt, dbtype,
             alpha_beta, r, tau, sigma, a0, N) %>%
    mutate(data = default,
           dbfrac = 100 * (1 - 1 / (1 + debt))) %>%
    arrange(data) %>%
    mutate(q = 1:n() / n()) %>%
    ungroup() %>%
    ggplot(aes(q, data,
               linetype = dbtype,
               color = interaction(100 * equity, eqtype))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(alpha_beta ~ dbfrac,
               scales = "free") +
    theme_tufte() +
    labs(x = "Quantile",
         y = "# defaults",
         color = "Equity fraction and type",
         linetype = "Debt type")

ggsave("/tmp/rand_default.pdf")


df <- read_csv("/tmp/baz.csv")

df %>%
    ggplot(aes(shock, loss,
               linetype = dbtype,
               color = interaction(equity, eqtype))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(alpha_beta ~ debt,
               scales = "free") +
    theme_tufte()
