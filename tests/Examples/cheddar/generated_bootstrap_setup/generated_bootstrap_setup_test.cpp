#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "UserInterface.h"
#include "extension/BootContext.h"

using word = uint64_t;
using UI = cheddar::UserInterface<word>;

void bootstrap__configure(std::shared_ptr<cheddar::BootContext<word>>& boot_ctx,
                          std::unique_ptr<UI>& ui);

TEST(CheddarGeneratedBootstrapSetupE2E, RealScaleSnuSetup) {
  std::shared_ptr<cheddar::BootContext<word>> boot_ctx;
  std::unique_ptr<UI> ui;
  bootstrap__configure(boot_ctx, ui);

  ASSERT_NE(boot_ctx, nullptr);
  ASSERT_NE(ui, nullptr);
  EXPECT_EQ(boot_ctx->param_.default_encryption_level_,
            boot_ctx->boot_param_.GetStCStartLevel());
  EXPECT_EQ(boot_ctx->boot_param_.GetEndLevel(),
            boot_ctx->param_.default_encryption_level_ - 2);
  EXPECT_GE(boot_ctx->boot_param_.GetEndLevel(), 0);
  EXPECT_NO_THROW(ui->GetEvkMap());
}
